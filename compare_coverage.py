"""Re-select saved DBSCAN grid results at different coverage floors; no refitting."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def review_run(folder, thresholds):
    metrics = pd.read_csv(folder / 'metrics.csv')
    required = {'run_key', 'coverage', 'n_clusters', 'dbcv', 'eps', 'min_samples'}

    if not required.issubset(metrics.columns):
        raise ValueError(f'Missing columns: {required - set(metrics.columns)}')
    if metrics['run_key'].duplicated().any():
        raise ValueError('Duplicate run_key in metrics')
    if not metrics['coverage'].between(0, 1).all():
        raise ValueError('Invalid coverage values')
    
    summary_path = folder / 'summary.json'
    experiment = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    details = experiment.get('representation', {})
    common = dict(run_dir=str(folder), dimensions=(details.get('shape') or [None, None])[1],
                  text_column=details.get('text_column', 'unknown'))
    
    candidates = []
    archive_path = folder / 'assignments.npz'
    if not archive_path.exists():
        raise FileNotFoundError('assignments.npz is needed to measure cluster dominance')
    
    with np.load(archive_path, allow_pickle=False) as saved:
        for _, r in metrics.iterrows():
            labels = saved[str(r['run_key'])]
            if labels.ndim != 1 or len(labels) == 0 or np.any((labels != -1) & (labels <= 0)):
                raise ValueError(f'Invalid labels for {r["run_key"]}')
            
            assigned = labels[labels > 0]
            _, sizes = np.unique(assigned, return_counts=True)
            sizes = np.sort(sizes)[::-1]
            if len(sizes) != r['n_clusters'] or not np.isclose(len(assigned)/len(labels), r['coverage'], atol=1e-8, rtol=0):
                raise ValueError(f'Metrics/assignments disagree for {r["run_key"]}')
            
            largest = int(sizes[0]) if len(sizes) else 0
            record = dict(**common, run_key=r['run_key'], eps=r['eps'], min_samples=int(r['min_samples']),
                          n_clusters=int(r['n_clusters']), coverage_pct=100*float(r['coverage']),
                          noise_pct=100*(1-float(r['coverage'])), dbcv=r['dbcv'],
                          silhouette=r.get('silhouette', np.nan), n_samples=len(labels),
                          largest_cluster_n=largest,
                          largest_cluster_pct_all=100*largest/len(labels),
                          largest_cluster_pct_assigned=100*largest/len(assigned) if len(assigned) else np.nan,
                          second_largest_n=int(sizes[1]) if len(sizes)>1 else 0,
                          median_cluster_n=float(np.median(sizes)) if len(sizes) else np.nan)
            
            for threshold in thresholds:
                record[f'eligible_coverage_{threshold:g}'] = bool(r['coverage'] >= threshold and r['n_clusters'] >= 2 and np.isfinite(r['dbcv']))
            candidates.append(record)

    table = pd.DataFrame(candidates)

    chosen = []
    for threshold in thresholds:
        eligible = table.loc[table[f'eligible_coverage_{threshold:g}']]
        if eligible.empty:
            chosen.append(dict(**common, min_coverage=threshold, status='no_eligible_configuration'))
        else:
            best = eligible.sort_values(['dbcv','coverage_pct','eps','min_samples'],ascending=[False,False,True,True]).iloc[0].to_dict()
            chosen.append(dict(**best, min_coverage=threshold, status='selected'))

    return chosen, candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--root', type=Path, help='Search this experiment directory recursively for metrics.csv')
    group.add_argument('--runs', nargs='+', type=Path, help='Specific experiment folders')
    p.add_argument('--thresholds', nargs='+', type=float, default=[.7, .5])
    p.add_argument('--output', type=Path, default=Path('coverage_review'))
    a=p.parse_args()

    if not all(0 <= t <= 1 for t in a.thresholds): p.error('Thresholds must be between 0 and 1')
    if a.output.exists() and any(a.output.iterdir()): p.error('Choose an empty output directory')
    folders = sorted({x.parent for x in a.root.rglob('metrics.csv')}) if a.root else a.runs
    if not folders: p.error('No saved experiment metrics found')
    a.output.mkdir(parents=True, exist_ok=True)

    selected, candidates, errors = [], [], []

    for folder in folders:
        try:
            ss, cc = review_run(folder, list(dict.fromkeys(a.thresholds)))
            selected.extend(ss); candidates.extend(cc)
            print(f'Reviewed {folder}: {len(cc)} saved configurations')
        except (ValueError, KeyError, OSError) as error:
            errors.append(dict(run_dir=str(folder), error=str(error)))
            print(f'Skipped {folder}: {error}')
    if selected: pd.DataFrame(selected).to_csv(a.output/'selected.csv', index=False)
    if candidates: pd.DataFrame(candidates).to_csv(a.output/'candidates.csv', index=False)
    if errors: pd.DataFrame(errors).to_csv(a.output/'errors.csv', index=False)

    (a.output/'README.txt').write_text(
        'Selection: maximize finite DBCV with at least two clusters and coverage >= threshold. '
        'Tie-break: higher coverage, lower eps, lower min_samples. No outcome labels used.\n'
        '50% minimum coverage allows UP TO 50% noise; it does not force 50% noise or smaller clusters.\n'
        'largest_cluster_pct_assigned measures dominance among assigned points; '
        'largest_cluster_pct_all includes noise in the denominator. Both are diagnostics, not selection constraints.\n'
        'No embeddings, distances, clustering, or quality metrics were recomputed. '
        'Original experiment outputs and selection metadata were not modified.\n'
        'candidates.csv contains the full saved grid with structural diagnostics. '
        'selected.csv contains the best eligible configuration for each threshold and run.\n',encoding='utf-8')
    print(f'Done: {len(selected)} selection rows, {len(errors)} skipped experiments. Outputs: {a.output}')

    if not selected: raise SystemExit(1)


if __name__=='__main__':main()
