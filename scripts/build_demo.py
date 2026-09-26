"""Build the public demo: an assets bundle for the Hugging Face model repo, plus a Docker variant.

build/demo/
    assets/        release bundles, gallery samples, results, model card -> HF model repo
    app.py, defect_detection/, requirements.txt, Dockerfile
                   self-contained container (any Docker host); uses assets/ locally

Streamlit Community Cloud runs app/streamlit_app.py straight from GitHub and downloads
assets/ from the Hub, so weights and dataset images never enter the Git repository.

    python scripts/build_demo.py
    cd build/demo && streamlit run app.py                     # test exactly what is shipped
    hf upload <user>/<assets-repo> build/demo/assets .         # publish assets
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from defect_detection.data.dataset import ANOMALOUS, NORMAL, expected_mask_path
from defect_detection.evaluation.metrics import condition_of
from defect_detection.release import LICENCE, export_release
from defect_detection.runs import resolve_run_dir, save_json
from defect_detection.utils.config import load_yaml, parse_data_config
from defect_detection.utils.logging import configure_logging, get_logger

logger = get_logger("build_demo")
REPO_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = """\
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user MPLCONFIGDIR=/tmp/matplotlib PORT=8501
EXPOSE 8501
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --browser.gatherUsageStats=false
"""

MODEL_CARD = """\
---
license: cc-by-nc-sa-4.0
tags:
  - anomaly-detection
  - image-segmentation
  - industrial-inspection
  - mvtec-ad-2
---

# Industrial defect inspection: demo assets

Release bundles (weights, thresholds, reference statistics, metrics) and gallery images for the
interactive demo of {github_url}.

| Bundle | Model | Supervision |
|---|---|---|
{rows}

Every metric in these bundles comes from tracked experiments; see the GitHub README for the
protocol and results. Anomaly scores are unnormalised and are not probabilities.

**Licence.** {licence} The gallery images are taken from the MVTec AD 2 public test set.
Dataset: Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised
Anomaly Detection*, arXiv:2503.21622, 2025.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--device", default="auto", help="device used to compute release reference statistics")
    return parser.parse_args()


def copy_samples(cfg: dict, data_cfg, assets: Path) -> list[dict]:
    """Copy gallery images (and masks) and describe them for the app."""
    samples_dir = assets / "samples"
    samples_dir.mkdir()
    entries = []
    for rel in cfg["samples"]:
        src = data_cfg.dataset.category_dir / rel
        label = ANOMALOUS if f"/{data_cfg.dataset.layout.bad_dir}/" in f"/{rel}" else NORMAL
        name = f"{'defective' if label == ANOMALOUS else 'normal'}_{src.stem}.png"
        shutil.copy2(src, samples_dir / name)
        entry = {"file": f"samples/{name}", "label": label, "condition": condition_of(src.name), "source": rel, "mask": None}
        if label == ANOMALOUS:
            split = next(k for k, v in data_cfg.dataset.layout.splits.items() if rel.startswith(f"{v}/"))
            mask = expected_mask_path(data_cfg.dataset, split, src)
            shutil.copy2(mask, samples_dir / f"{Path(name).stem}_mask.png")
            entry["mask"] = f"samples/{Path(name).stem}_mask.png"
        entries.append(entry)
    return entries


def build_assets(cfg: dict, data_cfg, assets: Path, device: str) -> None:
    rows = []
    releases = []
    for rel in cfg["releases"]:
        # Always use this machine's dataset copy (runs may record another machine's path, e.g. Colab).
        bundle = export_release(resolve_run_dir(rel["run"]), assets / "releases", rel["name"],
                                str(data_cfg.dataset.root), device)
        releases.append(rel["name"])
        meta = load_yaml(bundle / "config.yaml")["model"]
        rows.append(f"| `releases/{rel['name']}` | {meta['name']} | {rel.get('supervision', '')} |")
        logger.info("Exported release %s from %s", rel["name"], rel["run"])

    save_json(assets / "samples.json", copy_samples(cfg, data_cfg, assets))
    results = REPO_ROOT / cfg["results_file"]
    if results.is_file():
        shutil.copy2(results, assets / "results.json")
    save_json(assets / "demo.json", {"releases": releases, "github_url": cfg["github_url"], "licence": LICENCE})
    (assets / "README.md").write_text(
        MODEL_CARD.format(github_url=cfg["github_url"], rows="\n".join(rows), licence=LICENCE)
    )


def build_container(out: Path) -> None:
    shutil.copytree(REPO_ROOT / "src" / "defect_detection", out / "defect_detection",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO_ROOT / "app" / "streamlit_app.py", out / "app.py")
    shutil.copy2(REPO_ROOT / "app" / "requirements.txt", out / "requirements.txt")
    (out / "Dockerfile").write_text(DOCKERFILE)


def main() -> None:
    configure_logging()
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    data_cfg = parse_data_config(load_yaml(config_path.parent / cfg["data_config"]))

    out = Path(cfg["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    build_assets(cfg, data_cfg, out / "assets", args.device)
    build_container(out)
    size_mb = sum(f.stat().st_size for f in (out / "assets").rglob("*") if f.is_file()) / 1e6
    logger.info("Demo built in %s (assets: %.0f MB)", out, size_mb)


if __name__ == "__main__":
    main()
