# Industrial Visual Defect Detection & Anomaly Segmentation

Benchmarking deep-learning approaches to industrial visual quality control on
[MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) under
different labelling regimes (no defect labels, image-level labels, pixel-level labels).

> **Status:** Phase 1 (foundation) in progress. No results yet. The results
> table will be generated from tracked experiments once they exist.

## Local setup

Requires Python 3.11+. Developed on an Apple M2 (MPS backend).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Dataset

MVTec AD 2 requires registration and licence acceptance, so it is downloaded
manually and never committed. Place each category under `data/mvtec_ad_2/`
(or anywhere else, then pass `--data-root`):

```text
data/mvtec_ad_2/<category>/
├── train/good/*.png
├── validation/good/*.png
└── test_public/
    ├── good/*.png
    ├── bad/*.png
    └── ground_truth/bad/*_mask.png
```

The layout (split directory names, mask suffix, and so on) is configurable in
[configs/dataset.yaml](configs/dataset.yaml). `test_private` and
`test_private_mixed` have no public ground truth and are not used for local evaluation.

Validate a downloaded category and write a summary of measured dataset
statistics to `artifacts/data/`:

```bash
python scripts/prepare_data.py --category <category>
python scripts/visualise_samples.py --category <category> --split test
```

## Project layout

```text
configs/                 YAML configuration (data, models)
src/defect_detection/    library code (data, models, training, evaluation, inference, utils)
scripts/                 command-line entry points
tests/                   targeted tests (synthetic fixtures, no dataset required)
app/                     Streamlit demo (later phase)
artifacts/               generated outputs (git-ignored)
```
