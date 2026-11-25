"""Write the final, standalone handoff from verified stage receipts."""
import json
from pathlib import Path
from statistics import mean, median

root=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root/'evidence'/('checkpoint-transfer-'+name+'.json')).read_text())

confirm,wide,bpe,components,extra=[read(n) for n in ('confirm','width','bpe-confirm','components','escalate')]
out=['# Checkpoint-only transfer: findings and handoff','',
     'Everything in this report is DEV. The published starting point is d48b2df. The local Gates 30–32 branch was not in the available Git history. Imported skills are selected explicitly. Donor training and reference collection run in isolated producer processes. The converter reads static tensors/vocabulary, and the receiver learns only from ground-truth bytes. No teacher cache or teacher logits enter the optimizer.',
     '', '## What crossed','',
     'The two independently fine-tuned donors start from one parent and solve incompatible secret cue/byte rules. Both swaps use identical queries, vocabulary and receiver code. The first test uses fresh byte surfaces on previously learned key/cue combinations. The Markdown test measures all 256 next-byte probabilities on a file separate from training and validation. All old-mode tests use disjoint Python files.',
     '', '| Interface | Cases | Near before tuning | Near after stopping | Mean zero-update top-1 | Worst zero-update KL (bits) |',
     '|---|---:|---:|---:|---:|---:|']
winner=wide['method']
def row(title,arms):
    out.append(f"| {title} | {len(arms)} | {sum(r['zero']['near'] for r in arms)} | {sum(r['selected']['near'] for r in arms)} | {mean(r['zero']['agreement'] for r in arms):.2%} | {max(r['zero']['kl_bits'] for r in arms):.7f} |")
for key,domain in [('rules','rule'),('text','Markdown')]:
    row('32 → 32 '+domain,[r for case in confirm[key] for r in case['runs'] if r['method']==winner and r['arm']=='import'])
for domain in ('rule','text'):
    row('32 → 64 '+domain,[c['lift'] for r in wide['runs'] if r['domain']==domain for c in r['cases'].values()])
bw=bpe['winner']
ba=[r['arms'][bw] for r in bpe['runs']]
out+=['',f"Multi-byte BPE donor → raw-byte receiver: {bpe['near_cases']}/5 near-lossless after validation-selected stopping; mean zero-update agreement {mean(a['zero']['agreement'] for a in ba):.2%}. Best bridge: {bw}. This comparator uses a canonical-prefix-boundary byte posterior. Equal hidden dimensions alone do not qualify this interface.",
      '','### What a single weight-group swap changed','',
      '| Replaced from opposite donor | Own-rule accuracy | Opposite-rule accuracy | Cases meeting the stand-alone carrier criterion |',
      '|---|---:|---:|---:|']
for group in components['runs'][0]['scores']:
    s=[r['scores'][group] for r in components['runs']]
    out.append(f"| {group} | {mean(v['own']['accuracy'] for v in s):.1%} | {mean(v['opposite']['accuracy'] for v in s):.1%} | {sum(v['opposite']['accuracy']>=.90 and v['opposite']['kl_bits']<=.02 for v in s)}/{len(s)} |")
out+=['','## Time to the common task target','',
      'Includes checkpoint loading, conversion, tuning and validation. Excludes common recipient pretraining and independent auditing. Only pairs meeting both validation and test targets are included in savings. A censored run establishes no exact saving. Small-model wall times are local to this single-thread CPU workflow. Medians of savings are taken pairwise.',
      '','| Challenge | Comparator | Matched / total | Median receiver s | Median comparator s | Median s saved | Median % saved |',
      '|---|---|---:|---:|---:|---:|---:|']
def times(label,reference,pairs):
    good=[(a['total'],b['total']) for a,b in pairs if a['hit'] and b['hit']]
    if good:
        s=[b-a for a,b in good]
        pct=[1-a/b for a,b in good]
        out.append(f'| {label} | {reference} | {len(good)}/{len(pairs)} | {median(a for a,b in good):.4f} | {median(b for a,b in good):.4f} | {median(s):.4f} | {median(pct):.1%} |')
    else:
        out.append(f'| {label} | {reference} | 0/{len(pairs)} | — | — | censored | — |')
def timed(r,field):
    return dict(total=r[field],hit=r.get('matched',r.get('matched_target')))
for k in ('rules','text'):
    for base in ('scratch','parent'):
        pairs=[]
        for case in confirm[k]:
            for a in case['runs']:
                if a['method']==winner and a['arm']=='import':
                    b=next(r for r in case['runs'] if r['label']==a['label'] and r['arm']==base)
                    pairs.append(tuple(timed(r,'total_receiver_seconds') for r in (a,b)))
        times('32 → 32 '+k,base,pairs)
for k in ('rule','text'):
    for base in ('scratch','parent'):
        cases=[c for r in wide['runs'] if r['domain']==k for c in r['cases'].values()]
        times('32 → 64 '+k,base,[(timed(c['lift'],'total_seconds'),timed(c[base],'total_seconds')) for c in cases])
cost=confirm['summary']['rule'][winner]['matched_times']
out+=['',f"If creating the compatible 32-wide donor purely for one receiver, median net time saved on the {len(cost)} scratch-matched rule pairs was {median(r['new_donor_net_seconds_saved'] for r in cost):.4f} seconds. Producer creation and reference scoring are counted. An existing donor is the use case in the main timing table.",
      '','## Limits and disposition','',
      'The winning correction keeps quantized information about the omitted dense weights and computes an additional dense linear operation at inference. The original compatible screen measured a 179,859-byte imported skill and a 150,613-byte donor checkpoint. Closer predictions did not imply a smaller model. The wider transfer embeds a 32-dimensional function into a 64-dimensional repeated-coordinate subspace and increases storage again. None of these assays establishes broad knowledge transfer across unrelated model families.',
      '',f"The independently seeded FP16 fallback reached zero-update near-losslessness on {extra['summary']['rule']['residual16']['zero_near']}/10 rules and {extra['summary']['text']['zero_near']}/5 Markdown cases. Its complete joint time/fidelity gate was {extra['all_pass']}.",
      '', '## Continue or reproduce','',
      '1. Clone the complete research branch (the corpus reader uses the historical base Git object):',
      '   `git clone --branch dev/checkpoint-transfer-fidelity https://github.com/spoonman136668-ai/NexusExpanse.git`',
      '2. Use Python 3.12, CPU PyTorch 2.10.0, NumPy and pytest. Entry points, workflow invocations, and thresholds are documented in CHECKPOINT_TRANSFER_PROTOCOL.md, CHECKPOINT_BPE_PREREG.md and CHECKPOINT_WIDTH_PREREG.md. CHECKPOINT_TRANSFER_EXECUTION.md records the cold-reference correction.',
      '3. The seven JSON files under evidence/checkpoint-transfer-* are the machine-readable receipts (screen, confirmation, precision fallback, BPE screen, BPE confirmation, component swaps, width). Each includes the measured source commit. The component, BPE and width tests use separate fresh seeds. Keep test results out of training and winner selection.',
      '4. For the ready-made receiver-only demonstration, download the checkpoint-transfer-width-demo workflow artifact, extract it and run `python benchmarks/checkpoint_width_lift.py demo <directory-containing-example.json>`. A successful run verifies two cold restored skills and both original-route hashes and prints the two different outputs for one raw prefix. To rebuild the kit from source, run `python benchmarks/checkpoint_width_lift.py results/width-replay.json results/demo` after creating results/.',
      '5. The highest-value open problem is preserving causal histories across different symbol boundaries. Use the fixed 8-bit correction as a control, fresh donor pairs and an independent corpus. Measure full-distribution divergence, donor-swap specificity, held-out retention, payload, inference cost, and time to identical performance. Avoid claiming an exact time saving for a baseline that failed to hit the target.']
(root/'docs/CHECKPOINT_TRANSFER_FINDINGS.md').write_text('\n'.join(out)+'\n')
