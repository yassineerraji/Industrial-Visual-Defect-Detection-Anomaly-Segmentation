from defect_detection.evaluation.report import README_END, README_START, render_markdown, update_readme


def _row(**overrides):
    row = {
        "model": "autoencoder", "supervision": "normal_only", "protocol": None, "image_auroc": None,
        "image_ap": None, "f1": None, "pixel_auroc": None, "dice": None, "iou": None,
        "latency_ms": None, "checkpoint_mb": None, "peak_rss_mb": None,
    }
    row.update(overrides)
    return row


def test_missing_values_render_as_tbd_never_invented():
    table = render_markdown([_row()])
    assert table.count("TBD") == 10
    assert "Autoencoder" in table and "Normal only" in table


def test_present_values_are_formatted():
    table = render_markdown([_row(image_auroc=0.91234, latency_ms=12.345)])
    assert "0.912" in table and "12.3" in table


def test_readme_update_only_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")
    assert update_readme(readme, "NEW\n")
    assert readme.read_text() == f"intro\n{README_START}\nNEW\n{README_END}\noutro\n"
    plain = tmp_path / "plain.md"
    plain.write_text("no markers")
    assert not update_readme(plain, "x")
