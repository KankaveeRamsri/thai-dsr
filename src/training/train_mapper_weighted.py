"""Fresh weighted-layer mapper experiment. Separate outputs; never reads test audio."""
import argparse
import csv
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
from torch.nn.utils.rnn import pad_sequence

from src.models.weighted_layer_mapper import WeightedLayerMapper
from src.models.encoder import Wav2Vec2ContentEncoder
from src.models.mapper import build_mapper_from_config
from src.training.train import set_seed, masked_l1_loss, build_optimizer, build_scheduler, step_scheduler
from src.utils.mel import compute_mel
from src.inference.run import Generator, AttrDict, compute_mel_frame_count, MEL_CLAMP_MIN, MEL_CLAMP_MAX
from src.evaluation.metrics import load_audio_16k, compute_stoi, compute_pesq

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/checkpoints/mapper_layer9.pt'
VOC = ROOT/'vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000'


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def record(p, data):
    s = json.dumps(data, ensure_ascii=False, allow_nan=False)
    with p.open('a') as f: f.write(s+'\n')
    print(p.stem.upper(), s, flush=True)


def read_splits():
    splits = json.loads((ROOT/'data/splits_w5.json').read_text())['splits']
    ids = sum([splits[s] for s in ('train', 'val', 'test')], [])
    assert len(ids) == len(set(ids)), 'Split overlap'
    with (ROOT/'data/manifest_w5.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['severity'] == 'severe']
    by_id = {r['utterance_id']: r for r in rows}
    assert len(rows) == len(by_id) and set(ids) == set(by_id)
    # Match original dataset row order, then shuffle training rows each epoch.
    return {s: [r for r in rows if r['utterance_id'] in set(splits[s])] for s in splits}


def weight_stats(model):
    w = model.layer_weights.detach().cpu().numpy().astype(float)
    entropy = float(-(w*np.log(w)).sum())
    return dict(weights=w.tolist(), entropy=entropy, effective_layers=math.exp(entropy),
                max_layer=int(w.argmax()), max_weight=float(w.max()))


def main(args):
    os.chdir(ROOT)
    if min(args.max_steps, args.validation_interval, args.val_items, args.threads) < 1:
        raise ValueError('Counts must be positive')
    out = (ROOT/args.output_dir).resolve()
    allowed = ROOT/'results/checkpoints/weighted_layer_mapper'
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Use a NEW child of results/checkpoints/weighted_layer_mapper')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    set_seed(42)
    device = torch.device(args.device)
    baseline = torch.load(BASE, map_location='cpu', weights_only=True)
    cfg, optim_cfg = baseline['model_config'], baseline['train_config']['optim']
    split = read_splits()
    rows = split['train']+split['val']
    selected = np.linspace(0, len(split['val'])-1, min(args.val_items, len(split['val'])), dtype=int)
    val_audio = [split['val'][i] for i in selected]
    immutable = [BASE, VOC, VOC.with_name('config.json'), ROOT/'src/models/mapper.py',
                 ROOT/'data/manifest_w5.csv', ROOT/'data/splits_w5.json']
    hashes = {str(p): digest(p) for p in immutable}
    estimate = sum(sf.info(ROOT/r['distorted_path']).duration for r in rows)*50*25*1024*2
    if shutil.disk_usage(ROOT).free < estimate + 2*1024**3:
        raise RuntimeError('Insufficient disk: need estimated FP16 cache plus 2 GiB reserve')
    provenance = dict(args=vars(args), model_config=cfg, optim=optim_cfg, seed=42,
                      hashes=hashes, cache_estimated_bytes=estimate,
                      split_ids={s:[r['utterance_id'] for r in rr] for s,rr in split.items()},
                      audio_val_ids=[r['utterance_id'] for r in val_audio], test_used=False,
                      initialization='fresh mapper; zero logits/uniform weights',
                      alignment='training/val L1: clean mel length, like original; audio inference: distorted length',
                      cache='float16 all 25 hidden_states at native encoder rate; sum in float32',
                      layer_definition='hidden_states[0] pre-transformer representation; [1:25] transformer outputs')
    (out/'run.json').write_text(json.dumps(provenance, indent=2))
    cache_dir = (ROOT/args.cache_dir).resolve()
    if not cache_dir.is_relative_to(ROOT/'data/embeddings'):
        raise ValueError('Cache must be under data/embeddings')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = dict(model=cfg['encoder']['content_model'], revision='3155938c549b23eee16b1d4b55dcb161b7fe4bcf',
                      manifest_sha256=hashes[str(ROOT/'data/manifest_w5.csv')], dtype='float16', layers=25)
    meta_file = cache_dir/'metadata.json'
    if meta_file.exists():
        assert json.loads(meta_file.read_text()) == cache_meta, 'Cache provenance differs'
    else: meta_file.write_text(json.dumps(cache_meta, indent=2))
    targets, frames, cache_paths = {}, {}, {}
    encoder = None
    for i, row in enumerate(rows, 1):
        uid = row['utterance_id']; source = ROOT/row['distorted_path']
        key = hashlib.sha256((uid+digest(source)).encode()).hexdigest()
        dest = cache_dir/(key+'.npy'); cache_paths[uid] = dest
        if not dest.exists():
            if encoder is None:
                encoder = Wav2Vec2ContentEncoder(device=torch.device(args.encoder_device), layer=9)
                encoder.model.eval().requires_grad_(False)
                assert encoder.model.config._commit_hash == cache_meta['revision']
                encoder_before = {k: v._version for k,v in encoder.model.state_dict(keep_vars=True).items()}
            layers = np.stack(encoder.encode_all_layers(str(source))).astype(np.float16)
            assert layers.ndim == 3 and layers.shape[0] == 25 and layers.shape[2] == 1024
            if not np.isfinite(layers).all(): raise RuntimeError('Non-finite cache')
            temp = dest.with_suffix('.tmp')
            with temp.open('wb') as f: np.save(f, layers)
            os.replace(temp, dest)
        clean, sr = sf.read(ROOT/row['clean_path'], dtype='float32', always_2d=True)
        targets[uid] = torch.from_numpy(compute_mel(clean.mean(axis=1), sr=sr).T.copy())
        frames[uid] = compute_mel_frame_count(source)
        if i == 1 or i % 10 == 0 or i == len(rows): print(f'CACHE {i}/{len(rows)}', flush=True)
    if encoder is not None:
        assert encoder_before == {k:v._version for k,v in encoder.model.state_dict(keep_vars=True).items()}
        assert all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
        del encoder
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    (out/'cache_audit.json').write_text(json.dumps(dict(encoder_frozen=True, files=len(cache_paths),
        bytes=sum(p.stat().st_size for p in cache_paths.values()), metadata=cache_meta), indent=2))

    def batch(rr):
        hs = [torch.from_numpy(np.load(cache_paths[r['utterance_id']])).to(device) for r in rr]
        ts = [targets[r['utterance_id']].to(device) for r in rr]
        lengths = torch.tensor([len(t) for t in ts], device=device)
        return hs, pad_sequence(ts, batch_first=True), lengths

    # Reset after encoder construction: reproducible fresh mapper initialization.
    set_seed(42)
    model = WeightedLayerMapper(cfg).to(device)
    optimizer = build_optimizer(model, optim_cfg)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
    scheduler = build_scheduler(optimizer, optim_cfg['lr_scheduler'], optim_cfg['num_epochs'])
    old = build_mapper_from_config(cfg).to(device)
    old.load_state_dict(baseline['model_state_dict']); old.eval().requires_grad_(False)
    assert not torch.equal(model.mapper.lstm.weight_ih_l0.cpu(), old.lstm.weight_ih_l0.cpu())
    del baseline
    voc = Generator(AttrDict(json.loads(VOC.with_name('config.json').read_text()))).to(device)
    voc.load_state_dict(torch.load(VOC, map_location='cpu', weights_only=True)['generator'])
    voc.remove_weight_norm(); voc.eval().requires_grad_(False)
    voc_before = {k: v._version for k,v in voc.state_dict(keep_vars=True).items()}

    @torch.no_grad()
    def validation(step, use_old=False, audio=True):
        model.eval(); total = 0.
        for start in range(0, len(split['val']), 4):
            rr = split['val'][start:start+4]; hs, target, lengths = batch(rr)
            if use_old:
                ee = [torch.nn.functional.interpolate(h[9].float().T[None], size=int(n), mode='linear', align_corners=False)[0].T for h,n in zip(hs,lengths)]
                pred = old(pad_sequence(ee,batch_first=True), lengths)
            else: pred = model(hs, lengths)
            total += float(masked_l1_loss(pred, target, lengths))*len(rr)
        result = dict(step=step, val_mel_l1=total/len(split['val']))
        if audio:
            folder=out/'validation'/('baseline' if use_old else f'step_{step:06d}'); folder.mkdir(parents=True)
            items=[]
            for row in val_audio:
                uid=row['utterance_id']; hs,_,_=batch([row]); n=frames[uid]
                if use_old:
                    emb=torch.nn.functional.interpolate(hs[0][9].float().T[None],size=n,mode='linear',align_corners=False).transpose(1,2)
                    pred=old(emb)
                else: pred=model(hs,torch.tensor([n],device=device))
                wav=voc(pred.transpose(1,2).clamp(MEL_CLAMP_MIN,MEL_CLAMP_MAX)).squeeze().cpu().numpy()
                assert np.isfinite(wav).all()
                dest=folder/(uid+'.wav'); sf.write(dest,np.clip(wav,-1,1),22050,subtype='PCM_16')
                ref,est=load_audio_16k(ROOT/row['clean_path']),load_audio_16k(dest)
                item=dict(utterance_id=uid,stoi=float(compute_stoi(ref,est)))
                try: item['pesq']=float(compute_pesq(ref,est))
                except Exception as e: item.update(pesq=None,pesq_error=str(e))
                items.append(item)
            pesqs=[v['pesq'] for v in items if v['pesq'] is not None]
            result.update(stoi=float(np.mean([v['stoi'] for v in items])),pesq=float(np.mean(pesqs)) if pesqs else None,
                          pesq_valid=len(pesqs), items=items)
        result.update(weight_stats(model) if not use_old else {})
        model.train()
        return result

    baseline_result=validation(0,use_old=True)
    record(out/'baseline.jsonl',baseline_result)
    del old;gc.collect()
    trend=[validation(0)];record(out/'validation.jsonl',trend[-1])
    bs=int(optim_cfg['batch_size']); generator=torch.Generator().manual_seed(42)
    losses=[]; times=[]; step=0; epoch=0; best=float('inf'); min_weight_grad=float('inf'); max_weight_grad=0.
    while step < args.max_steps:
        epoch+=1; order=torch.randperm(len(split['train']),generator=generator).tolist()
        for start in range(0,len(order),bs):
            step+=1; begin=time.perf_counter()
            rr=[split['train'][i] for i in order[start:start+bs]]
            hs,target,lengths=batch(rr); optimizer.zero_grad(set_to_none=True)
            pred=model(hs,lengths);loss=masked_l1_loss(pred,target,lengths)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward()
            # Audit only: no clipping, matching the original optimization.
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            wg=float(model.layer_logits.grad.norm());min_weight_grad=min(min_weight_grad,wg);max_weight_grad=max(max_weight_grad,wg)
            if wg == 0:raise RuntimeError('Missing layer-weight gradient')
            optimizer.step();losses.append(float(loss.detach()));times.append(time.perf_counter()-begin)
            entry=dict(step=step,epoch=epoch,mel_l1=losses[-1],gradient_norm=float(norm),weight_gradient_norm=wg,seconds=times[-1],lr=optimizer.param_groups[0]['lr'])
            with (out/'losses.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
            if step==1 or step%10==0:print('TRAIN',json.dumps(entry),flush=True)
            if step%args.validation_interval==0 or step==args.max_steps:
                result=validation(step);trend.append(result);record(out/'validation.jsonl',result)
            if step>=args.max_steps:break
        # Original ReduceLROnPlateau cadence: once per completed epoch, all val rows.
        val=validation(step,audio=False);record(out/'epochs.jsonl',dict(epoch=epoch,**val))
        if start+bs>=len(order):step_scheduler(scheduler,optim_cfg['lr_scheduler'],val['val_mel_l1'])
        if val['val_mel_l1']<best:
            best=val['val_mel_l1'];torch.save(dict(step=step,model=model.state_dict(),model_config=cfg),out/'best.pt')
    torch.save(dict(step=step,epoch=epoch,model=model.state_dict(),model_config=cfg,optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),provenance=provenance),out/'final.pt')
    assert hashes == {str(p):digest(p) for p in immutable}
    assert voc_before == {k:v._version for k,v in voc.state_dict(keep_vars=True).items()}
    assert all(not p.requires_grad and p.grad is None for p in voc.parameters())
    summary=dict(steps=step,epochs=epoch,all_finite=True,first_50_loss=float(np.mean(losses[:50])),last_50_loss=float(np.mean(losses[-50:])),
                 mean_step_seconds=float(np.mean(times)),weight_grad_min=min_weight_grad,weight_grad_max=max_weight_grad,
                 baseline=baseline_result,validation=trend,final_weights=weight_stats(model),best_val_l1=best,
                 immutable_files_unchanged=True,vocoder_frozen=True,test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    print('FINISHED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--cache-dir',default='data/embeddings/weighted_all25_fp16')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--encoder-device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--validation-interval',type=int,default=100)
    p.add_argument('--val-items',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    main(p.parse_args())
