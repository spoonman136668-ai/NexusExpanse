"""DEV-only projected plasticity using the common longitudinal comparison harness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.consolidation_dev import main

ARMS=('frozen','shared_no_decay','project8','project16','sparse_support')
if __name__=='__main__':
    main(ARMS, 'nexus-projected-plasticity-dev-v1')
