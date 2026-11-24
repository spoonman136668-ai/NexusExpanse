# Nexus Expanse

Independent research prototype: raw-byte causal attention, DirectMonarch FFNs, uncertainty-gated computation, episodic suffix memory and local fast weights. This is a byte-level attention architecture, not a non-transformer model. Do not integrate into other projects without separate authorization and qualification.

## Baseline custody

Original archive SHA-256: `241b0c365a2d46eea49e06cf5337f06798250b7561de5ccce9574c59f684119e`.
Original import: `bb22c229aebae264d5b85fcc5e5a16901bbfeae8`, tree `fce3c3d7fd89f2996a15bfbc4685115353552b62`.
`gate1_results.json` is imported historical evidence, not a reproduced result. The original benchmark uses an unavailable local Yggdrasil path and an external hard-coded dense score. It is retained unchanged.

## Engineering

Python with PyTorch and pytest. Run `python -m pytest -q`. The new qualification preflight only checks explicit train/validation byte files and emits their SHA-256 identities; it does not train or consume a held-out scientific trial.

`python benchmarks/qualification_preflight.py --train PATH --validation PATH --output NEW.json`

## Research status

See `docs/RESEARCH.md`. No checkpoint, frozen source corpus, independently reproduced dense control or scientific first-use authorization is present. Keep original evidence immutable. No deployment, external runner, or scheduled execution authorized by this import.
