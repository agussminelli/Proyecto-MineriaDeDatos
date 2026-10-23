# Visual examples

Run from the repository root. Existing dependencies are sufficient.

```powershell
# Geometry: fitting, core/border points, noise and unseen assignments
py -m examples.dbscan_demo

# Text: 13 public synthetic phrases, no embedding download
py -m examples.text_demo --dataset synthetic

# Text: reuse one saved verbal-autopsy partition without refitting
py -m examples.text_demo --dataset verbal-autopsy `
    --run-dir "experiments\2026-10-21\17-52-39-553828" `
    --eps 0.725 --min-samples 5
```

Adjust the saved-run path if you moved it. Missing artifacts produce an error
with the synthetic command; the dataset is never silently substituted.
Outputs default to `outputs/examples/`. Use `--output-dir` to choose another
folder. Repeating a command replaces its figures. No narrative text is exported.
PCA is only a display projection; clustering uses the original features.
