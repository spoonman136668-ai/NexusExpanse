#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
base=bb21fbe53c0299d432e866eb4d016c25b81ee6eb
mkdir -p results
git fetch --quiet origin "$base"
git worktree add --quiet --detach "$RUNNER_TEMP/nexus-base" "$base"
python - "$RUNNER_TEMP/nexus-base" <<'PY'
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

def measure(root):
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, 'benchmarks/prefix_causality_dev.py'],
                                cwd=root, text=True, capture_output=True)
        duration = time.perf_counter() - start
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        data = json.loads(result.stdout)
        assert result.returncode == int(bool(data['fails']))
        if samples and data != samples[-1]['sweep']:
            raise RuntimeError('inconsistent deterministic sweep')
        samples.append(dict(seconds=duration, sweep=data))
    return dict(sweep=data, seconds=[r['seconds'] for r in samples],
                median_seconds=statistics.median(r['seconds'] for r in samples))

before = measure(sys.argv[1])
after = measure('.')
evidence = dict(schema='nexus-prefix-causality-dev-v2', qualification='DEVELOPMENT_ONLY',
                baseline=before, repaired=after, platform=platform.platform(),
                protocol='Original 100-seed CPU sweep, 3 fresh processes per revision, wall time includes startup.',
                passed=not after['sweep']['fails'])
Path('results/prefix-causality-dev-002.json').write_text(json.dumps(evidence, indent=2)+'\n')
assert evidence['passed']
PY
python -m pytest -q | tee results/regression.txt
cp results/prefix-causality-dev-002.json evidence/
git add evidence/prefix-causality-dev-002.json
git commit -m 'Gate 1: record measured causal-prefix repair evidence'
git push origin "HEAD:refs/heads/$GITHUB_REF_NAME"

python benchmarks/ablation_dev.py --output results/ablation-dev-002.json
python benchmarks/plasticity_dev.py --output results/plasticity-dev-002.json
cp results/{ablation-dev-002,plasticity-dev-002}.json evidence/
git add evidence/ablation-dev-002.json evidence/plasticity-dev-002.json
git commit -m 'Gate 2: requalify matched ablation and projected plasticity on repaired backbone'
git push origin "HEAD:refs/heads/$GITHUB_REF_NAME"
