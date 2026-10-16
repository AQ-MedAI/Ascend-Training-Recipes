#!/usr/bin/env bash
# Reconstruct the public K26 training baseline. Internal adaptations are not included.
set -euo pipefail
python3 - <<'FETCH'
import json
import subprocess
from pathlib import Path
root = Path('/opt/ascend-training')
manifest = json.loads((root / 'manifest/k26-sources.json').read_text())
for name, source in manifest['sources'].items():
    target = root / name
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', source['repository']], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth=1', 'origin', source['commit']], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != source['commit']:
        raise RuntimeError(f'Unexpected source revision: {name}: {actual}')
# Source-only stage: installation/import validation requires the pending K26 dependency lock.
(root / 'manifest/k26-build-status.json').write_text(json.dumps({
    'stage': 'public-training-sources', 'runtime_validated': False,
    'unresolved': manifest['unresolved'],
}, indent=2) + '\n')
FETCH
