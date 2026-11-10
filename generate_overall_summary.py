import sys
from pathlib import Path
sys.path.insert(0, r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-verify\src')

from analysis.tables import generate_summary_json
import json

artifacts_root = Path('artifacts')
datasets = ['sst2', 'imdb', 'wikitext2']

overall = {
    'datasets': {},
    'summary': {},
}

for ds in datasets:
    summary_path = artifacts_root / ds / 'stats' / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            overall['datasets'][ds] = json.load(f)
    else:
        print(f"Warning: No summary for {ds}")

# Aggregate pass/fail
claims = ['claim_performance', 'claim_grad_1_over_s', 'claim_head_collapse', 'claim_early_diagnostic']
for claim in claims:
    passes = sum(1 for ds in datasets if ds in overall['datasets'] and overall['datasets'][ds].get(claim) == 'pass')
    total = sum(1 for ds in datasets if ds in overall['datasets'])
    overall['summary'][claim] = f'{passes}/{total} pass'

# Write overall summary JSON
with open(artifacts_root / 'OVERALL_SUMMARY.json', 'w') as f:
    json.dump(overall, f, indent=2)

# Generate markdown
md = ['# Overall Summary\n', '| Dataset | Task | Performance | Grad 1/s | Head Collapse | Early Diag |']
md.append('|---------|------|-------------|----------|---------------|------------|')
for ds in datasets:
    if ds in overall['datasets']:
        d = overall['datasets'][ds]
        md.append(f'| {ds} | {d.get("task","n/a")} | {d.get("claim_performance","n/a")} | {d.get("claim_grad_1_over_s","n/a")} | {d.get("claim_head_collapse","n/a")} | {d.get("claim_early_diagnostic","n/a")} |')

md.append('\n## Aggregate')
for claim, val in overall['summary'].items():
    md.append(f'- **{claim}**: {val}')

md.append('\n## Per-Dataset Details')
for ds in datasets:
    if ds in overall['datasets']:
        d = overall['datasets'][ds]
        md.append(f'\n### {ds.upper()}')
        md.append(f'- Task: {d["task"]}')
        md.append(f'- Metric values: standard={d["test_or_val_metric"]["standard"]:.4f}, temp={d["test_or_val_metric"]["temp"]:.4f}, dk={d["test_or_val_metric"]["dk"]:.4f}')
        md.append(f'- Notes: {d.get("notes","")}')

with open(artifacts_root / 'OVERALL_SUMMARY.md', 'w') as f:
    f.write('\n'.join(md))

print('Overall summary written to artifacts/OVERALL_SUMMARY.md')
print('\n'.join(md))