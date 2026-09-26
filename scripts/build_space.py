"""Assemble the self-contained Hugging Face Space (Docker + Streamlit) for the public demo.

The output folder is exactly what gets deployed; test it locally first:
    python scripts/build_space.py
    cd build/hf_space && streamlit run app.py
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

logger = get_logger("build_space")
REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.0
torchvision==0.29.0
numpy==2.4.6
pillow==12.3.0
pyyaml==6.0.3
matplotlib==3.11.2
streamlit==1.64.0
"""

DOCKERFILE = """\
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8501
# XSRF protection blocks uploads inside the Hugging Face iframe.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \\
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
"""

SPACE_README = """\
---
title: Industrial Defect Inspection
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: cc-by-nc-sa-4.0
short_description: Anomaly detection & defect localisation on MVTec AD 2
---

# Industrial Visual Defect Inspection

Interactive demo of a benchmark comparing anomaly-detection approaches under
different labelling regimes on MVTec AD 2 (`sheet_metal`).

Source code, protocol and full results: {github_url}

{licence} Dataset: Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced
Scenarios for Unsupervised Anomaly Detection*, arXiv:2503.21622, 2025.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto", help="device used to compute release reference statistics")
    return parser.parse_args()


def copy_samples(cfg: dict, data_cfg, out: Path) -> list[dict]:
    """Copy gallery images (and masks) and describe them for the app."""
    samples_dir = out / "samples"
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


def main() -> None:
    configure_logging()
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    data_cfg = parse_data_config(load_yaml(config_path.parent / cfg["data_config"]), data_root=args.data_root)

    out = Path(cfg["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(REPO_ROOT / "src" / "defect_detection", out / "defect_detection",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO_ROOT / "app" / "streamlit_app.py", out / "app.py")

    releases = []
    for rel in cfg["releases"]:
        # Always use this machine's dataset copy (runs may record another machine's path, e.g. Colab).
        bundle = export_release(
            resolve_run_dir(rel["run"]), out / "releases", rel["name"], str(data_cfg.dataset.root), args.device
        )
        releases.append(rel["name"])
        logger.info("Exported release %s from %s", bundle.name, rel["run"])

    save_json(out / "samples.json", copy_samples(cfg, data_cfg, out))
    results = REPO_ROOT / cfg["results_file"]
    if results.is_file():
        shutil.copy2(results, out / "results.json")
    save_json(out / "demo.json", {"releases": releases, "github_url": cfg["github_url"], "licence": LICENCE})

    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "README.md").write_text(SPACE_README.format(github_url=cfg["github_url"], licence=LICENCE))
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n")
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    logger.info("Space built in %s (%.0f MB)", out, size_mb)


if __name__ == "__main__":
    main()
