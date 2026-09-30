"""Bounded mel-only LSGAN fine-tuning of the trained layer-9 mapper.

The frozen universal vocoder is invoked ONLY inside no-grad validation.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.models.mapper import build_mapper_from_config
from src.models.encoder import Wav2Vec2ContentEncoder
from src.models.mel_discriminator import (MelDiscriminator, adversarial_weight,
    score_mels, discriminator_loss, generator_loss)
from src.training.train import set_seed, masked_l1_loss
from src.training.grad_clip import clip
from src.training.train_mapper_weighted import read_splits, digest, record, ROOT, BASE, VOC
from src.utils.mel import compute_mel
from src.inference.run import Generator, AttrDict, compute_mel_frame_count, MEL_CLAMP_MIN, MEL_CLAMP_MAX
from src.evaluation.metrics import load_audio_16k, compute_stoi, compute_pesq
from src.evaluation.mapper_eval import audio_metrics, mel_l1, w5_eval_rows

W5_MANIFEST, W5_SPLITS = 'data/manifest_w5.csv', 'data/splits_w5.json'
ENCODER_REVISION = '3155938c549b23eee16b1d4b55dcb161b7fe4bcf'


def interpolate(h, n):
    return F.interpolate(h.T[None], size=int(n), mode='linear', align_corners=False)[0].T


def main(args):
    os.chdir(ROOT)
    if min(args.max_steps, args.validation_interval, args.threads,
           args.checkpoint_every, args.eval_every or 1) < 1:
        raise ValueError('Steps, intervals and threads must be positive')
    if args.warmup < 0 or args.ramp < 1 or args.warmup + args.ramp >= args.max_steps:
        raise ValueError('Smoke must exercise warmup, ramp and full-weight phases')
    if any(not math.isfinite(v) or v <= 0 for v in (args.mel_weight,args.adv_weight,args.mapper_lr,args.discriminator_lr)):
        raise ValueError('Weights and learning rates must be finite and positive')
    out = (ROOT/args.output_dir).resolve()
    allowed = ROOT/'results/checkpoints/mel_adversarial_mapper'
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Use a new child of results/checkpoints/mel_adversarial_mapper')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); set_seed(42)
    device = torch.device('cpu')
    init = ROOT/args.init_checkpoint if args.init_checkpoint else BASE
    baseline = torch.load(init, map_location='cpu', weights_only=True)
    cfg = baseline['model_config']; split = read_splits(manifest=args.manifest, splits_path=args.splits)
    # Audio validation always uses the 8 fixed W5 reference clips (same as before on W5).
    w5_val, val_audio = w5_eval_rows()
    val_ids = {r['utterance_id'] for r in split['val']}
    assert {r['utterance_id'] for r in w5_val} <= val_ids, 'W5 val utterances must stay in val'
    w5_manifest = (ROOT/args.manifest).resolve() == (ROOT/W5_MANIFEST).resolve()
    immutable = [init,VOC,VOC.with_name('config.json'),ROOT/'src/models/mapper.py',
                 ROOT/args.manifest,ROOT/args.splits]
    hashes = {str(p):digest(p) for p in immutable}
    model = build_mapper_from_config(cfg).to(device)
    model.load_state_dict(baseline['model_state_dict'])
    assert all(torch.equal(v, baseline['model_state_dict'][k]) for k,v in model.state_dict().items())
    del baseline
    discriminator = MelDiscriminator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mapper_lr, betas=(0.9,0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5,0.999))
    provenance = dict(args=vars(args), seed=42, model_config=cfg, hashes=hashes,
        initialization=f'exact {init.relative_to(ROOT) if init.is_relative_to(ROOT) else init} model_state_dict; fresh mel discriminator and Adam states',
        discriminator_parameters=sum(p.numel() for p in discriminator.parameters()),
        audio_val_ids=[r['utterance_id'] for r in val_audio],
        split_ids={s:[r['utterance_id'] for r in rr] for s,rr in split.items()},
        layer=9, encoder_frozen=True, vocoder_training=False, test_used=False,
        alignment='original: clean length for training/L1, distorted length for audio inference',
        loss='mel_weight * masked L1 + ramped adv_weight * mean((D(pred)-1)^2)',
        discriminator_loss='0.5 * mean((D(real)-1)^2 + D(pred.detach())^2); true-length utterances')
    (out/'run.json').write_text(json.dumps(provenance, indent=2))
    print('PROCESS',json.dumps(dict(pid=os.getpid(),session=os.getsid(0),process_group=os.getpgrp())),flush=True)
    # W5: the all-layer fp16 cache (unchanged). Other manifests: the float32
    # layer-9 cache written by src.preprocessing.extract_embedding. Misses are encoded.
    cache = ROOT/'data/embeddings/weighted_all25_fp16'
    if w5_manifest:
        meta = json.loads((cache/'metadata.json').read_text())
        assert meta == dict(model=cfg['encoder']['content_model'], revision=ENCODER_REVISION,
                           manifest_sha256=hashes[str(ROOT/W5_MANIFEST)],dtype='float16',layers=25)
    embeddings, targets, frames = {}, {}, {}
    encoder = None
    rows = split['train']+split['val']
    for i,row in enumerate(rows,1):
        uid=row['utterance_id']; source=ROOT/row['distorted_path']
        if w5_manifest:
            key=hashlib.sha256((uid+digest(source)).encode()).hexdigest()
            path=cache/(key+'.npy')
        else:
            path=ROOT/'data/embeddings/distorted'/f'{uid}_severe_layer09.npy'
        if path.exists():
            embedding=(np.load(path,mmap_mode='r')[9] if w5_manifest else np.load(path)).astype(np.float32)
        else:
            if encoder is None:
                encoder=Wav2Vec2ContentEncoder(device=device,layer=9)
                encoder.model.eval().requires_grad_(False)
                assert encoder.model.config._commit_hash == ENCODER_REVISION
            embedding=encoder.encode(str(source)).astype(np.float16).astype(np.float32)
        assert embedding.ndim==2 and embedding.shape[1]==1024 and np.isfinite(embedding).all()
        embeddings[uid]=torch.from_numpy(embedding)
        clean,sr=sf.read(ROOT/row['clean_path'],dtype='float32',always_2d=True)
        targets[uid]=torch.from_numpy(compute_mel(clean.mean(axis=1),sr=sr).T.copy())
        frames[uid]=compute_mel_frame_count(source)
        if i==1 or i%30==0 or i==len(rows): print(f'CACHE {i}/{len(rows)}',flush=True)
    if encoder is not None:
        assert all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
    del encoder;gc.collect()
    voc=Generator(AttrDict(json.loads(VOC.with_name('config.json').read_text()))).to(device)
    voc.load_state_dict(torch.load(VOC,map_location='cpu',weights_only=True)['generator'])
    voc.remove_weight_norm();voc.eval().requires_grad_(False)
    voc_before={k:v._version for k,v in voc.state_dict(keep_vars=True).items()}
    assert not ({id(p) for p in model.parameters()} & {id(p) for p in discriminator.parameters()})
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
    assert {id(p) for g in d_optimizer.param_groups for p in g['params']} == {id(p) for p in discriminator.parameters()}

    def batch(rows):
        ts=[targets[r['utterance_id']] for r in rows]
        lengths=torch.tensor([len(t) for t in ts])
        hs=[interpolate(embeddings[r['utterance_id']],n) for r,n in zip(rows,lengths)]
        return pad_sequence(hs,batch_first=True),pad_sequence(ts,batch_first=True),lengths

    losses=[]
    @torch.no_grad()
    def validation(step, audio=True):
        model.eval();discriminator.eval();total=dl=gl=0.
        for start in range(0,len(split['val']),4):
            rr=split['val'][start:start+4];hs,target,lengths=batch(rr)
            pred=model(hs,lengths)
            total+=float(masked_l1_loss(pred,target,lengths))*len(rr)
            real=score_mels(discriminator,target,lengths);fake=score_mels(discriminator,pred,lengths)
            dl+=float(discriminator_loss(real,fake))*len(rr)
            gl+=float(generator_loss(fake))*len(rr)
        count=len(split['val'])
        result=dict(step=step,val_mel_l1=total/count,discriminator_loss=dl/count,
            generator_adversarial_loss=gl/count,adversarial_weight=adversarial_weight(step,args.warmup,args.ramp,args.adv_weight))
        for name in ('mapper','discriminator'):
            result[name+'_clip_count']=sum(r[name+'_grad_norm']>1 for r in losses)
            result[name+'_clip_rate']=result[name+'_clip_count']/len(losses) if losses else 0.
            result[name+'_max_grad_norm']=max((r[name+'_grad_norm'] for r in losses),default=0.)
        if audio:
            folder=out/'validation'/f'step_{step:06d}';folder.mkdir(parents=True)
            items=[]
            for row in val_audio:
                uid=row['utterance_id'];h=interpolate(embeddings[uid],frames[uid])[None]
                pred=model(h,torch.tensor([frames[uid]]))
                wav=voc(pred.transpose(1,2).clamp(MEL_CLAMP_MIN,MEL_CLAMP_MAX)).squeeze().cpu().numpy()
                assert np.isfinite(wav).all()
                dest=folder/(uid+'.wav');sf.write(dest,np.clip(wav,-1,1),22050,subtype='PCM_16')
                ref,est=load_audio_16k(ROOT/row['clean_path']),load_audio_16k(dest)
                item=dict(utterance_id=uid,stoi=float(compute_stoi(ref,est)))
                try:item['pesq']=float(compute_pesq(ref,est))
                except Exception as e:item.update(pesq=None,pesq_error=str(e))
                items.append(item)
            pesqs=[v['pesq'] for v in items if v['pesq'] is not None]
            result.update(stoi=float(np.mean([v['stoi'] for v in items])),pesq=float(np.mean(pesqs)) if pesqs else None,
                          pesq_valid=len(pesqs),items=items)
        model.train();discriminator.train()
        return result

    trend=[validation(0)];record(out/'validation.jsonl',trend[-1])
    sampler=torch.Generator().manual_seed(42);step=epoch=0
    best=trend[0]['val_mel_l1']
    def checkpoint(name):
        torch.save(dict(step=step,epoch=epoch,model_state_dict=model.state_dict(),model_config=cfg,
            discriminator=discriminator.state_dict(),optimizer=optimizer.state_dict(),d_optimizer=d_optimizer.state_dict(),
            rng_state=torch.get_rng_state(),sampler_rng=sampler.get_state(),provenance=provenance),out/name)
    checkpoint('best.pt')

    def periodic_checkpoint():
        (out/'checkpoints').mkdir(exist_ok=True)
        name=f'checkpoints/step_{step:06d}.pt'
        checkpoint(name)
        shutil.copyfile(out/name,out/'checkpoints/latest.pt.tmp')
        os.replace(out/'checkpoints/latest.pt.tmp',out/'checkpoints/latest.pt')

    def data(row):
        uid=row['utterance_id']
        return embeddings[uid],targets[uid],frames[uid]

    def periodic_eval():
        """mel L1 over the 32 W5 val utterances; STOI/PESQ on the 8 W5 reference clips."""
        model.eval()
        mel=mel_l1(model,w5_val,data)
        audio=audio_metrics(model,voc,val_audio,data,out/'eval'/f'step_{step:06d}')
        model.train()
        entry=dict(step=step,mel_l1=mel,stoi=audio['stoi'],pesq=audio['pesq'])
        with (out/'eval_log.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        print('EVAL',json.dumps(entry),flush=True)
    while step<args.max_steps:
        epoch+=1;order=torch.randperm(len(split['train']),generator=sampler).tolist()
        for start in range(0,len(order),4):
            step+=1;begin=time.perf_counter()
            weight=adversarial_weight(step,args.warmup,args.ramp,args.adv_weight)
            hs,target,lengths=batch([split['train'][i] for i in order[start:start+4]])
            optimizer.zero_grad(set_to_none=True);d_optimizer.zero_grad(set_to_none=True)
            pred=model(hs,lengths)
            dloss=discriminator_loss(score_mels(discriminator,target,lengths),score_mels(discriminator,pred.detach(),lengths))
            if not torch.isfinite(dloss):raise RuntimeError('Non-finite D loss')
            dloss.backward()
            dn,dp=clip(discriminator.parameters(), diagnostics=out/'gradients.jsonl',
                       step=step, component='discriminator', weight=weight)
            d_optimizer.step()
            d_optimizer.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            mel=masked_l1_loss(pred,target,lengths)
            adv=generator_loss(score_mels(discriminator,pred,lengths))
            loss=args.mel_weight*mel+weight*adv
            if not torch.isfinite(loss):raise RuntimeError('Non-finite mapper loss')
            loss.backward()
            gn,gp=clip(model.parameters(), diagnostics=out/'gradients.jsonl',
                       step=step, component='mapper', weight=weight)
            optimizer.step()
            assert all(p.grad is None for p in discriminator.parameters())
            discriminator.requires_grad_(True)
            entry=dict(step=step,epoch=epoch,mel_l1=float(mel.detach()),total_loss=float(loss.detach()),
                discriminator_loss=float(dloss.detach()),generator_adversarial_loss=float(adv.detach()),adversarial_weight=weight,
                mapper_grad_norm=gn,mapper_post_clip_norm=gp,discriminator_grad_norm=dn,discriminator_post_clip_norm=dp,
                seconds=time.perf_counter()-begin)
            losses.append(entry)
            with (out/'losses.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
            if step==1 or step%10==0:print('TRAIN',json.dumps(entry),flush=True)
            if step%args.validation_interval==0 or step==args.max_steps:
                result=validation(step);trend.append(result);record(out/'validation.jsonl',result)
                checkpoint(f'step_{step:06d}.pt')
            if step%args.checkpoint_every==0 or step==args.max_steps:periodic_checkpoint()
            if step%(args.eval_every or args.checkpoint_every)==0 or step==args.max_steps:periodic_eval()
            if step>=args.max_steps:break
        result=validation(step,audio=False);record(out/'epochs.jsonl',dict(epoch=epoch,**result))
        if result['val_mel_l1']<best:best=result['val_mel_l1'];checkpoint('best.pt')
    checkpoint('final.pt')
    assert hashes=={str(p):digest(p) for p in immutable}
    assert voc_before=={k:v._version for k,v in voc.state_dict(keep_vars=True).items()}
    assert all(not p.requires_grad and p.grad is None for p in voc.parameters())
    summary=dict(steps=step,all_finite=True,first_50_mel_l1=float(np.mean([r['mel_l1'] for r in losses[:50]])),
        last_50_mel_l1=float(np.mean([r['mel_l1'] for r in losses[-50:]])),validation=trend,
        mean_step_seconds=float(np.mean([r['seconds'] for r in losses])),best_val_l1=best,
        immutable_files_unchanged=True,vocoder_frozen=True,encoder_frozen=True,test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('FINISHED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--init-checkpoint',default=None,
                   help='mapper checkpoint to start from (default: results/checkpoints/mapper_layer9.pt)')
    p.add_argument('--manifest',default=W5_MANIFEST)
    p.add_argument('--splits',default=W5_SPLITS)
    p.add_argument('--checkpoint-every',type=int,default=300,
                   help='also save checkpoints/step_N.pt and checkpoints/latest.pt every N steps')
    p.add_argument('--eval-every',type=int,default=None,
                   help='append W5 mel L1 + 8-clip STOI/PESQ to eval_log.jsonl every N steps (default: --checkpoint-every)')
    p.add_argument('--validation-interval',type=int,default=100)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--ramp',type=int,default=100)
    p.add_argument('--mel-weight',type=float,default=10.)
    p.add_argument('--adv-weight',type=float,default=0.1)
    p.add_argument('--mapper-lr',type=float,default=1e-5)
    p.add_argument('--discriminator-lr',type=float,default=1e-4)
    main(p.parse_args())
