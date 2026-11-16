#!/usr/bin/env python3
"""Package the expanded-dataset baseline run for Colab.

Bundles code, configs, the expanded manifest/splits, every clean and severe
distorted WAV the manifest references (embeddings are recomputed on the GPU),
and the frozen UNIVERSAL_V1 vocoder for the W5-protocol comparison. Paths are
repository-relative; bundle_manifest.json inside the zip lists sizes/SHA-256.
"""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'scripts/colab_expanded_baseline_bundle.zip'
HIFIGAN = ['LICENSE', 'README.md', 'env.py', 'models.py', 'utils.py', 'meldataset.py',
           'config_v1.json', 'checkpoints/UNIVERSAL_V1/config.json',
           'checkpoints/UNIVERSAL_V1/g_02500000']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    files = sorted(p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts)
    files += sorted((ROOT / 'configs').glob('*.yaml'))
    files += [ROOT / p for p in (
        'requirements.txt', 'scripts/evaluate_expanded_mapper.py', 'scripts/build_expanded_splits.py',
        'data/manifest_expanded.csv', 'data/splits_expanded.json',
        'data/manifest_w5.csv', 'data/splits_w5.json')]
    files += [ROOT / 'vendor/hifi-gan' / p for p in HIFIGAN]
    with (ROOT / 'data/manifest_expanded.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            files += [ROOT / row['clean_path'], ROOT / row['distorted_path']]

    entries = []
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, rel)
            entries.append(dict(path=rel, bytes=path.stat().st_size, sha256=sha256(path)))
        zf.writestr('bundle_manifest.json', json.dumps(dict(files=entries), ensure_ascii=False, indent=1))
    wavs = [e for e in entries if e['path'].endswith('.wav')]
    print(f'{OUTPUT.relative_to(ROOT)}: {OUTPUT.stat().st_size:,} bytes, '
          f'{len(entries)} files ({len(wavs)} wav, {sum(e["bytes"] for e in entries):,} bytes uncompressed)')


if __name__ == '__main__':
    main()
