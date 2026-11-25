"""Render the continuing handoff from the committed evidence files."""
import json
from pathlib import Path
import statistics as st
import sys

root=Path(__file__).resolve().parents[1]
def read(stage):
    p=root/'evidence'/f'checkpoint-transfer-{stage}.json'
    return json.loads(p.read_text()) if p.exists() else None

screen,confirm,extra=map(read,('screen','confirm','escalate'))
out=['# Checkpoint transfer handoff','',
     'Isolated DEV series based on d48b2df. All decisions and thresholds were frozen in docs/CHECKPOINT_TRANSFER_PROTOCOL.md.',
     'The donor producer is a separate process. The importer accepts tensors and byte-row metadata only. Audit probabilities are created by the producer and never supplied to the importer or optimizer.',
     'The original Python route is frozen and explicitly selected. These results cover compatible attention geometry and 256 singleton-byte symbols. Both models perform neural inference when their own predictions are requested.',
     '', '## Recovery and reproduction','',
     'Use the source SHA in each JSON receipt. Run the workflow with workflow_dispatch on this branch to resume at the first absent receipt. On Linux CPU with Python 3.12, PyTorch 2.10.0+cpu, NumPy and pytest, run the stage commands in .github/workflows/checkpoint-transfer.yml. Keep earlier result files in evidence/ for deterministic selection. Incremental partial receipts are diagnostic; a restarted stage recomputes its whole frozen cohort. The action archives partial results and test logs on failure.',
     '', '## Screen','']
if screen:
    out+=['| Rank | Method | Worst KL, bits | Mean KL, bits | Skill checkpoint B |',
          '|---:|---|---:|---:|---:|']
    out += [f'| {i} | {m} | {w:.7f} | {a:.7f} | {n} |' for i,(w,a,n,m) in enumerate(screen['ranking'],1)]
for name,x in [('Confirmation',confirm),('Precision fallback',extra)]:
    if not x:
        continue
    out+=['',f'## {name}','','Source commit `'+x['source']+'`.','',
          '| Challenge | Method | Near at zero | Near after tuning | Qualified | Matched time pairs | Median time saved | Range |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for challenge,items in [('Rule',x['summary']['rule']),('Markdown',{next(iter(x['summary']['rule'])):x['summary']['text']})]:
        for method,s in items.items():
            times=s['matched_times']
            f=[r['fraction_saved']*100 for r in times]
            secs=[r['seconds_saved'] for r in times]
            stats=f'{st.median(secs):.4f} s ({st.median(f):.1f}%)' if f else 'censored'
            ran=f'{min(f):.1f}% to {max(f):.1f}%' if f else '—'
            out.append(f"| {challenge} | {method} | {s['zero_near']}/{s['cases']} | {s['selected_near']}/{s['cases']} | {s['qualified']}/{s['cases']} | {len(times)} | {stats} | {ran} |")
    out+=['','### Charging donor creation','']
    for label,s in [('rule',next(iter(x['summary']['rule'].values()))),('Markdown',x['summary']['text'])]:
        t=s['matched_times']
        if t:
            net=st.median(v['new_donor_net_seconds_saved'] for v in t)
            out.append(f"On {label}, median net seconds saved when building a new donor for one receiver: {net:.4f}. The above table assumes an existing checkpoint. Donor cost includes qualification and reference export. Base receiver preparation is common.")
    out+=['','### Donor attribution at zero receiver updates','',
          '| Method | Correct rule | Opposite rule | Own FFNs + opposite remainder, accuracy on own rule | Opposite FFNs + own remainder, accuracy on own rule |',
          '|---|---:|---:|---:|---:|']
    for m in x['summary']['rule']:
        scored=[r['zero'] for case in x['rules'] for r in case['runs'] if r['arm']=='import' and r['method']==m]
        hybrids=[(case['crossed_components'][label][m],label) for case in x['rules'] for label in ('A','B')]
        def hybrid(k):
            return st.mean(v[k][label]['accuracy'] for v,label in hybrids)
        out.append(f"| {m} | {st.mean(r['accuracy'] for r in scored):.1%} | {st.mean(r['opposite_accuracy'] for r in scored):.1%} | {hybrid('opposite_rest_own_ffn'):.1%} | {hybrid('own_rest_opposite_ffn'):.1%} |")
    out+=['',f"Exact cold restores of imported and old-mode predictions: {sum(c['cold_exact'] for c in x['rules']+x['text'])}.",
          'The text and rule evidence includes individual time traces, producer costs, checkpoint sizes, fingerprints and uncensored matched times. A missed target at the tuning ceiling establishes no exact time-to-target.']
out+=['','## Disposition','']
latest=extra or confirm
if latest:
    if latest['all_pass']:
        out+=['The leading method met this bounded series\'s fidelity, retention, attribution and timing criteria. Next qualify a donor with a genuinely different multi-byte vocabulary or geometry on an independently sourced corpus. Keep this positive control and report full parameter, disk, conversion and inference costs.']
    elif not extra:
        out+=['The locked precision fallback is next. Preserve the confirmation result.']
    else:
        out+=['The frozen ladder is exhausted. Preserve the failed criteria. Before tuning another method, identify the failed seed, comparator and criterion in the raw receipts. Distinguish static fidelity from measured time savings; an uncrossed scratch target leaves the savings censored.']
else:
    out+=['Confirmation is pending. Advance the first two rows of the frozen ranking, with the first row entering Markdown.']
(root/'docs'/'CHECKPOINT_TRANSFER_HANDOFF.md').write_text('\n'.join(out)+'\n')
