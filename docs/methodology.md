# Methodology and full results

The detailed companion to the [README](../README.md). Every number here comes from tracked
experiment artifacts (`artifacts/runs/<run>/*.json`, mirrored in MLflow and summarised in
[results/results.json](../results/results.json)).

## Motivation

Defects on production lines are rare, varied and expensive to annotate. A model that needs only
images of good parts can be deployed on day one; a model that needs pixel-accurate defect masks
may localise better but costs expert labelling time. The right choice depends on detection
quality, localisation quality, robustness to imaging drift, inference cost and labelling cost
together.

## Dataset

**MVTec AD 2, category `sheet_metal`** (licence CC BY-NC-SA 4.0, downloaded manually from MVTec,
never committed). Measured with `scripts/prepare_data.py`:

| Split | Normal | Anomalous | Resolution (W×H) |
|---|---:|---:|---|
| train | 137 | 0 | 4224×1056 |
| validation | 19 | 0 | 4224×1056 |
| test_public | 24 | 90 | 4224×1056 |

- Pixel masks exist only for `test_public`. They are strictly binary; defects cover on average
  0.58% of an anomalous image.
- `test_public` photographs **19 physical parts (15 defective, 4 good) under 6 acquisition
  conditions** each: `regular`, `overexposed`, `underexposed`, `shift_1..3`. All images of a part
  share the same defect.
- `test_private` / `test_private_mixed` have no public labels and are not used.

## Labelling regimes and models

| Regime | Labels used | Model | Idea |
|---|---|---|---|
| A | none (normal images only) | Convolutional autoencoder | Reconstruction error of a model trained on good parts |
| A | none (normal images only) | PatchCore (ResNet-18) | Nearest-neighbour distance to a memory bank of pretrained patch features |
| C | pixel masks | U-Net (ResNet-18 encoder) | Supervised per-pixel defect segmentation (BCE + Dice) |

## Experimental protocol

- **Preprocessing:** images resized to **256×1024**, which keeps the exact 1:4 aspect ratio.
  The resolution was fixed by rule (aspect ratio plus the memory budget), not tuned on test data.
  Thin scratches become 1–2 px wide at this resolution, a known limitation for all models.
- **Regime A (autoencoder, PatchCore):** fit on the 137 `train` images. The checkpoint
  (autoencoder: lowest validation reconstruction loss) and all thresholds are selected on the 19
  normal `validation` images:
  - image threshold = mean + 3·std of validation image scores;
  - pixel threshold = 99.9th percentile of validation pixel scores.

  Test data is used only for the final evaluation.
- **Regime C (U-Net):** pixel labels exist only in `test_public`, so there is no official
  labelled training split. The U-Net uses **5-fold grouped cross-validation over
  `test_public`**:
  - folds are grouped by physical part, so all 6 photos of a part fall in the same fold, and
    stratified to 3 defective parts per fold;
  - each fold model trains on the official normal `train` images plus the other 4 folds, for a
    fixed 30 epochs with no checkpoint selection;
  - thresholds are predefined at 0.5 on the sigmoid output.

  Each test image is scored by the model that never saw its part, and metrics are computed on
  the pooled out-of-fold predictions. The 114 images are the same as for Regime A, but the
  protocol differs: each U-Net fold model also trains on labelled images of the other test
  parts, including their over/underexposed and shifted versions. A final model trained on all
  labelled data is used for deployment; its expected performance is the cross-validation
  estimate.
- **Metrics:**
  - image level: AUROC, AP, precision, recall and F1 at the stored threshold, plus the
    false-positive rate (FPR) on normal parts;
  - pixel level: AUROC over all test pixels, plus Dice/IoU at the stored pixel threshold,
    computed at model resolution;
  - per-condition image AUROC for the six acquisition conditions.
- **Robustness:** brightness, contrast, Gaussian noise and blur at 3 fixed severities
  ([configs/robustness.yaml](../configs/robustness.yaml)), with thresholds kept at their
  validation values. Results are stored separately from the clean test metrics.
- **Operational:** single-image latency through the inference API (preprocessing + model + map
  upsampling to 4224×1056), checkpoint size and peak memory, at batch size 1 on an Apple M2
  (8 GB), on CPU and on MPS.

## Full results

The full table (AP, F1, FPR, CPU and MPS latency, checkpoint size, peak memory) is generated
into [results/results_table.md](../results/results_table.md).

### Key findings

- **The autoencoder fails on this texture** (image AUROC 0.43, below chance). Its
  reconstruction error follows the embossed surface pattern and the lighting rather than the
  defects. Thin scratches are invisible, and it collapses on shifted parts (AUROC 0.13–0.18 on
  `shift_1/2`).
- **PatchCore is the strongest model that needs no labels.** Image AUROC is 0.71, pixel AUROC
  0.86 and Dice 0.34, with no false alarms on normal parts at its validation-derived threshold,
  but it catches only 41% of defective images. It is stable across all six acquisition
  conditions (AUROC 0.67–0.75).
- **Pixel labels buy detection, at the cost of false alarms.** The U-Net has the best image
  AUROC (0.85, recall 0.84), but its fixed 0.5 threshold flags 29% of normal parts. With no
  labelled calibration data outside cross-validation, that threshold cannot be tuned without
  leakage.
- **Pixel AUROC misleads for the U-Net.** It outputs near-zero probabilities for most pixels,
  including missed defect pixels, so the ranking is mostly ties (pixel AUROC 0.59) even though
  its Dice (0.21) beats the autoencoder's. Dice/IoU is the fairer localisation comparison.
- **F1 misleads on this test set.** It is 79% defective, so a model that flags everything
  scores F1 = 0.88. Always read F1 together with the FPR.

### Robustness

- **PatchCore** loses ranking quality under brightness and contrast shifts (image AUROC falls
  from 0.71 to 0.53–0.55 at strong severity). Under strong noise or moderate-to-strong blur,
  its fixed threshold breaks down: the FPR rises to 0.58–1.00.
- **The U-Net** is nearly insensitive to brightness (AUROC 0.85–0.88), most likely because its
  training folds include over- and underexposed photos of other parts. It degrades under
  contrast, noise and blur (AUROC 0.64–0.75 at strong severity, with FPR up to 0.88).
- **The autoencoder's scores scale with intensity.** Darkening shrinks reconstruction errors
  until every image passes (F1 = 0).

Full results are in each run's `robustness_test.json` and are plotted in the demo app.

### Model selection

No model wins on every axis. **PatchCore is the default deployment choice** when no defect
labels exist:
- it needs only images of good parts;
- it raised no false alarms on the clean test set;
- it runs in about 0.37 s per image on CPU.

**The U-Net is worth its labelling cost only if missed defects are much more expensive than
false alarms**, and only once labelled data exists to calibrate its threshold. The autoencoder
is not suitable for this surface type.

## Demo app

The Streamlit app (`app/`) runs on Streamlit Community Cloud straight from this repository
([live demo](https://industrial-defect-inspection.streamlit.app)).
Model weights, thresholds and gallery images are not stored in Git: they are published as
release bundles to a Hugging Face model repo
([yassineerraji/industrial-defect-inspection-assets](https://huggingface.co/yassineerraji/industrial-defect-inspection-assets))
and downloaded at startup.

The app inspects sample parts or uploaded images with all three models side by side. It shows
PASS/FAIL against the stored thresholds, the anomaly score, a defect heatmap, the ground-truth
outline and latency. Gallery images come from the public test set:
- The autoencoder and PatchCore never trained on them.
- The U-Net scores each gallery image with the cross-validation fold model that never saw that
  part.

The app also shows the results, robustness curves, the protocol, and a session monitoring view
with an input-brightness drift check against the training data.

## Reproducing everything

Place the dataset under `data/mvtec_ad_2/<category>/` (layout configurable in
[configs/dataset.yaml](../configs/dataset.yaml), or pass `--data-root`), then:

```bash
python scripts/prepare_data.py --category sheet_metal          # validate layout, measure statistics
python scripts/visualise_samples.py --split test               # sample/mask figure

python scripts/train.py --config configs/autoencoder.yaml      # -> artifacts/runs/<run_id>/
python scripts/train.py --config configs/patchcore.yaml
python scripts/train.py --config configs/unet.yaml             # 5 fold models + final model (GPU recommended)

python scripts/evaluate.py --run <run_id>                      # test metrics + figures
python scripts/robustness.py --run <run_id>                    # perturbation study
python scripts/benchmark.py --run <run_id> --device cpu        # latency / memory (also --device mps)
python scripts/make_results_table.py --runs <ae> <patchcore> <unet>

mlflow ui --backend-store-uri sqlite:///mlflow.db              # browse tracked runs

python scripts/build_demo.py                                   # release bundles + Docker variant -> build/demo
cd build/demo && streamlit run app.py                          # run exactly what gets shipped
hf upload <user>/<assets-repo> build/demo/assets .             # publish assets (repo set in configs/demo.yaml)
python scripts/make_readme_figure.py                           # README figure
```

**Training on a cloud GPU.** The U-Net's 6 × 30 epochs take about 2 h and are too heavy for a
fanless laptop. [cloud/colab_train_unet.ipynb](../cloud/colab_train_unet.ipynb) runs the same
scripts on a free Colab GPU and saves the run directory to Google Drive. Runs are portable:
unzip into `artifacts/runs/` and pass `--data-root` to evaluate against a local copy of the
data. The reported U-Net run was trained on a Colab T4 and reproduced locally on MPS to 4
decimal places.

Every run directory is self-contained (`config.yaml`, weights, thresholds, history,
`run_info.json`, metrics). MLflow mirrors it; set `MLFLOW_TRACKING_URI` to track elsewhere, for
example in an Azure ML workspace.

## Project layout

```text
configs/                 YAML experiment configs (data, models, robustness, demo)
src/defect_detection/
  data/                  indexing, preprocessing, augmentation, grouped folds, validation
  models/                autoencoder, PatchCore, U-Net
  training/              training loop, losses, normal-only and cross-validated pipelines
  evaluation/            scoring, thresholds, metrics, robustness, benchmark, figures, report
  inference/             Predictor: predict(image) -> score, threshold, map, latency
  runs.py, tracking.py   run directories and MLflow mirroring
  release.py             deployable release bundles (weights, thresholds, reference statistics)
scripts/                 command-line entry points
app/                     Streamlit demo + its runtime requirements
cloud/                   Colab launcher for GPU training
tests/                   targeted tests on synthetic fixtures (no dataset needed)
```

## Limitations

- A single category (`sheet_metal`) so far.
- The U-Net protocol differs from Regime A (see above), so its results are not a like-for-like
  comparison. Its deployed all-data model has no independent test estimate; the
  cross-validation score stands in for it.
- The public test set has only 19 distinct physical parts (4 normal), so metrics have wide
  uncertainty; the FPR in particular rests on 24 normal images.
- Downscaling 4.125× makes hairline scratches very thin.
- No hyperparameter search was run (it would need labelled validation data that does not exist
  outside the test set), so every model uses fixed, a-priori settings.
- The autoencoder and PatchCore runs predate commit tracking (`git_commit` is null).
