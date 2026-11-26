"""Verify and transport a local tensor-only sharded checkpoint into a native skill."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_expanse.cli import main


if __name__ == "__main__":
    sys.exit(main(["assimilate-zero-exec", *sys.argv[1:]]))
