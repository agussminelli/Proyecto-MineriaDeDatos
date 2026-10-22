"""Plot summaries from benchmark_dbscan.py; run after a completed benchmark."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
with (args.results / 'summary.csv').open() as f:
    rows = list(csv.DictReader(f))
for field, ylabel, name in [
    ('median_seconds', 'Median elapsed time (seconds)', 'runtime.png'),
    ('median_process_peak_rss_mib', 'Median process peak RSS (MiB)', 'memory.png'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ['single_fit', 'parameter_sweep']):
        for variant in ['own_on_demand', 'own_cached', 'sklearn_brute']:
            subset = sorted([r for r in rows if r['scenario'] == scenario and r['variant'] == variant],
                            key=lambda r: int(r['n']))
            ax.plot([int(r['n']) for r in subset], [float(r[field]) for r in subset],
                    marker='o', label=variant)
        ax.set(title=scenario.replace('_', ' ').title(), xlabel='Number of samples', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(args.results / name, dpi=180)
    plt.close(fig)
