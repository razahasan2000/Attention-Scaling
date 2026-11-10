import sys
from pathlib import Path
sys.path.insert(0, r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-verify\src')

from analysis import analyze_runs, save_analysis_results, save_all_tables, generate_all_figures
from probes import run_all_probes_for_dataset
from data import get_config, DataCache, prepare_dataset
import torch

config = get_config('imdb')
device = torch.device(config['training'].get('device', 'cuda'))
cache = DataCache(config['data']['cache_dir'])
prepared = prepare_dataset('imdb', config, cache)

artifacts_dir = Path('artifacts/imdb')
results = analyze_runs(artifacts_dir / 'runs', 'classification')
save_analysis_results(results, artifacts_dir / 'stats')

probes = run_all_probes_for_dataset('imdb', config, prepared, [0,1], ['standard','temp','dk'], artifacts_dir, device)

save_all_tables(results, probes, artifacts_dir / 'stats', 'imdb')
generate_all_figures(results, probes, artifacts_dir / 'figures', 'classification')

print("IMDB analysis complete!")