"""Append the reproducible follow-up to the research handoff."""
import json
from pathlib import Path
import statistics as st

root=Path(__file__).resolve().parents[1]
path=root/'docs/CHECKPOINT_TRANSFER_HANDOFF.md'
title='## Multi-byte donor follow-up'
before=path.read_text().split(title)[0].rstrip()
screen=json.loads((root/'evidence/checkpoint-transfer-bpe-screen.json').read_text())
out=[title,'','Protocol: docs/CHECKPOINT_BPE_PREREG.md. Reproduce with benchmarks/checkpoint_bpe_challenge.py bpe-screen or bpe-confirm and a new JSON output path. The winner was chosen before independent confirmation. The three-seed screen ranking (worst donor KL bits) was '+', '.join(f'{r[-1]} {r[0]:.5f}' for r in screen['ranking'])+'.']
result=root/'evidence/checkpoint-transfer-bpe-confirm.json'
if result.exists():
    x=json.loads(result.read_text())
    out+=['',f"Winner: {x['winner']}; near-lossless on {x['near_cases']}/5 independent seeds.",
          '','| Seed | Zero KL bits | Zero agreement | Final agreement | Final KL bits | Test BPB (receiver / donor) | Matched seconds saved |',
          '|---:|---:|---:|---:|---:|---:|---:|']
    for r in x['runs']:
        a=r['arms'][x['winner']]
        b=r['arms']['scratch']
        total=(f"{b['total_seconds']-a['total_seconds']:.4f}" if a['matched'] and b['matched'] else 'threshold unconfirmed')
        out.append(f"| {r['seed']} | {a['zero']['kl_bits']:.4f} | {a['zero']['agreement']:.1%} | {a['selected']['agreement']:.1%} | {a['selected']['kl_bits']:.4f} | {a['selected']['bpb']:.4f} / {r['producer']['test']['bpb']:.4f} | {total} |")
    out+=['','All rates concern the canonical-prefix-boundary reference. No oracle distributions entered conversion or gradients. Retain both the compatible positive control and this harder challenge. For an independent next experiment, change one interface mismatch at a time and use fresh files and seeds.']
else:
    out+=['','Status: confirmation pending. The workflow resumes from the committed screen.']
components=root/'evidence/checkpoint-transfer-components.json'
if components.exists():
    runs=json.loads(components.read_text())['runs']
    out+=['','### Independent tensor-group swaps','',
          '| Group replaced by opposite donor | Own-rule accuracy | Opposite-rule accuracy | Opposite-donor KL bits |',
          '|---|---:|---:|---:|']
    for group in runs[0]['scores']:
        ms=[r['scores'][group] for r in runs]
        out.append(f"| {group} | {st.mean(v['own']['accuracy'] for v in ms):.1%} | {st.mean(v['opposite']['accuracy'] for v in ms):.1%} | {st.mean(v['opposite']['kl_bits'] for v in ms):.4f} |")
    out+=['','See evidence/checkpoint-transfer-components.json for all five fresh donor pairs and both swap directions.']
path.write_text(before+'\n\n'+'\n'.join(out)+'\n')
