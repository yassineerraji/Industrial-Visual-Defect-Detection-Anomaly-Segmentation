from defect_detection.evaluation.report import README_END, README_START, render_compact, render_markdown, update_readme


def _row(**overrides):
    row = {
        "model": "autoencoder", "supervision": "normal_only", "protocol": None, "image_auroc": None,
        "image_ap": None, "f1": None, "false_positive_rate": None, "pixel_auroc": None, "dice": None,
        "latency_cpu_ms": None, "latency_mps_ms": None, "checkpoint_mb": None, "peak_rss_cpu_mb": None,
    }
    row.update(overrides)
    return row


def test_missing_values_render_as_tbd_never_invented():
    table = render_markdown([_row()])
    assert table.count("TBD") == 11
    assert "Autoencoder" in table and "Normal only" in table


def test_present_values_are_formatted():
    table = render_markdown([_row(image_auroc=0.91234, latency_cpu_ms=12.345)])
    assert "0.912" in table and "12.3" in table


def test_compact_table_formats_and_never_invents():
    table = render_compact([_row(image_auroc=0.7125, false_positive_rate=0.0, latency_cpu_ms=367.3), _row()])
    assert "| Autoencoder | none | 0.71 | 0% | TBD | TBD | 0.37 s |" in table
    assert table.count("TBD") == 2 + 5  # row 1: pixel AUROC and Dice unset; row 2: nothing measured


def test_readme_update_only_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")
    assert update_readme(readme, "NEW\n")
    assert readme.read_text() == f"intro\n{README_START}\nNEW\n{README_END}\noutro\n"
    plain = tmp_path / "plain.md"
    plain.write_text("no markers")
    assert not update_readme(plain, "x")
