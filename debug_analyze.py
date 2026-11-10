import sys
from pathlib import Path
sys.path.insert(0, r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-verify\src')

from analysis import analyze_runs

artifacts_dir = Path('attention-scaling-verify/artifacts/sst2')
results = analyze_runs(artifacts_dir / 'runs', 'classification')
print(f"Results: {results}")