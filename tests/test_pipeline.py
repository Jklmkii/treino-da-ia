from pathlib import Path
import yaml
from PIL import Image

from src.dataset_prep import (
    CLASS_NAMES,
    create_stratified_subset,
    parse_label_classes,
)
from src.train_and_compare import (
    build_side_by_side_panel,
    generate_comparison_charts,
)
from src.build_docs import (
    build_colab_notebook,
    build_readme_markdown,
)


def _make_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 160, 80)).save(path)


def test_parse_label_classes_handles_empty_and_multiclass(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n   \n", encoding="utf-8")
    assert parse_label_classes(empty) == set()

    missing = tmp_path / "does_not_exist.txt"
    assert parse_label_classes(missing) == set()

    multi = tmp_path / "multi.txt"
    multi.write_text(
        "0 0.50 0.50 0.20 0.20\n3 0.30 0.30 0.15 0.15\n",
        encoding="utf-8",
    )
    assert parse_label_classes(multi) == {0, 3}


def test_create_stratified_subset_disjoint_and_absolute_yaml(tmp_path: Path) -> None:
    source_root = tmp_path / "raw_wildlife"
    for split, count_per_class in [("train", 5), ("val", 3)]:
        for cls_id in CLASS_NAMES:
            for idx in range(count_per_class):
                stem = f"{split}_cls{cls_id}_{idx}"
                _make_dummy_image(source_root / "images" / split / f"{stem}.jpg")
                label_file = source_root / "labels" / split / f"{stem}.txt"
                label_file.parent.mkdir(parents=True, exist_ok=True)
                # Add a second class on idx==0 to exercise multiclass deduplication
                extra = f"\n{(cls_id + 1) % 4} 0.2 0.2 0.1 0.1" if idx == 0 else ""
                label_file.write_text(
                    f"{cls_id} 0.5 0.5 0.3 0.3{extra}\n",
                    encoding="utf-8",
                )

    output_root = tmp_path / "subset_wildlife"
    yaml_path = create_stratified_subset(
        source_root=source_root,
        output_root=output_root,
        train_per_class=3,
        val_per_class=2,
        seed=42,
    )

    assert yaml_path.exists()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert Path(cfg["path"]).is_absolute()
    assert cfg["names"] == CLASS_NAMES

    train_imgs = list((output_root / "images" / "train").glob("*.jpg"))
    val_imgs = list((output_root / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) == 12
    assert len(val_imgs) == 8

    train_stems = {p.stem for p in train_imgs}
    val_stems = {p.stem for p in val_imgs}
    assert train_stems.isdisjoint(val_stems)

    # Every class must appear in both splits
    train_classes = set()
    for lbl in (output_root / "labels" / "train").glob("*.txt"):
        train_classes |= parse_label_classes(lbl)
    val_classes = set()
    for lbl in (output_root / "labels" / "val").glob("*.txt"):
        val_classes |= parse_label_classes(lbl)
    assert train_classes == {0, 1, 2, 3}
    assert val_classes == {0, 1, 2, 3}


def test_generate_comparison_charts_creates_pngs(tmp_path: Path) -> None:
    sample_summary = {
        "YOLOv8n": {
            "precision": 0.85,
            "recall": 0.80,
            "map50": 0.88,
            "map50_95": 0.65,
            "inference_ms": 18.2,
            "train_time_s": 120.0,
            "params_m": 3.01,
        },
        "YOLO11n": {
            "precision": 0.88,
            "recall": 0.83,
            "map50": 0.90,
            "map50_95": 0.68,
            "inference_ms": 15.4,
            "train_time_s": 115.0,
            "params_m": 2.59,
        },
    }
    paths = generate_comparison_charts(sample_summary, tmp_path)
    assert len(paths) >= 2
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_build_side_by_side_panel_combines_images(tmp_path: Path) -> None:
    img_a = Image.new("RGB", (320, 240), color=(200, 50, 50))
    img_b = Image.new("RGB", (320, 240), color=(50, 200, 50))
    img_c = Image.new("RGB", (320, 240), color=(50, 50, 200))
    out_file = tmp_path / "panel.jpg"
    saved = build_side_by_side_panel(
        images=[("Ground Truth", img_a), ("YOLOv8n", img_b), ("YOLO11n", img_c)],
        output_path=out_file,
        title="Comparativo — Elefante",
    )
    assert saved.exists()
    with Image.open(saved) as combined:
        assert combined.width >= 960
        assert combined.height > 240


def test_build_colab_notebook_creates_valid_ipynb(tmp_path: Path) -> None:
    import json

    sample_summary = {
        "YOLOv8n": {
            "precision": 0.86,
            "recall": 0.82,
            "map50": 0.89,
            "map50_95": 0.67,
            "inference_ms": 17.5,
            "train_time_s": 540.0,
            "params_m": 3.01,
            "per_class_ap50": {"buffalo": 0.90, "elephant": 0.88, "rhino": 0.87, "zebra": 0.91},
        },
        "YOLO11n": {
            "precision": 0.89,
            "recall": 0.85,
            "map50": 0.91,
            "map50_95": 0.70,
            "inference_ms": 14.8,
            "train_time_s": 490.0,
            "params_m": 2.59,
            "per_class_ap50": {"buffalo": 0.92, "elephant": 0.90, "rhino": 0.89, "zebra": 0.93},
        },
    }
    nb_path = tmp_path / "notebooks" / "treinando_yolo_african_wildlife.ipynb"
    saved = build_colab_notebook(sample_summary, nb_path, repo_slug="Jklmkii/treino-da-ia")
    assert saved.exists()
    nb_data = json.loads(saved.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] == 4
    all_sources = "\n".join("".join(c["source"]) for c in nb_data["cells"])
    assert "colab.research.google.com" not in all_sources
    code_sources = "\n".join(
        "".join(c["source"]) for c in nb_data["cells"] if c["cell_type"] == "code"
    )
    assert "from ultralytics import YOLO" in code_sources
    assert "yolov8n.pt" in code_sources
    assert "yolo11n.pt" in code_sources
    assert "african-wildlife.yaml" in code_sources


def test_build_readme_markdown_includes_boxes_and_comparison(tmp_path: Path) -> None:
    sample_summary = {
        "YOLOv8n": {
            "precision": 0.86,
            "recall": 0.82,
            "map50": 0.89,
            "map50_95": 0.67,
            "inference_ms": 17.5,
            "train_time_s": 540.0,
            "params_m": 3.01,
            "per_class_ap50": {"buffalo": 0.90, "elephant": 0.88, "rhino": 0.87, "zebra": 0.91},
        },
        "YOLO11n": {
            "precision": 0.89,
            "recall": 0.85,
            "map50": 0.91,
            "map50_95": 0.70,
            "inference_ms": 14.8,
            "train_time_s": 490.0,
            "params_m": 2.59,
            "per_class_ap50": {"buffalo": 0.92, "elephant": 0.90, "rhino": 0.89, "zebra": 0.93},
        },
    }
    readme_path = tmp_path / "README.md"
    build_readme_markdown(sample_summary, readme_path, repo_slug="Jklmkii/treino-da-ia")
    text = readme_path.read_text(encoding="utf-8")
    assert "colab.research.google.com" not in text
    assert "colab" not in text.lower()
    assert "p.p" not in text.lower()
    assert "porcentua" in text.lower()
    assert len(text) < 1800
    assert "YOLOv8n" in text and "YOLO11n" in text
    assert "results/predictions/comparison_elephant.jpg" in text
    assert "results/charts/metrics_comparison.png" in text


def test_end_to_end_generated_deliverables_exist_and_valid() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    metrics_json = repo_root / "results" / "metrics_comparison.json"
    assert metrics_json.exists()
    summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    for model_name in ("YOLOv8n", "YOLO11n"):
        assert model_name in summary
        assert summary[model_name]["map50"] > 0.70

    for species in ("buffalo", "elephant", "rhino", "zebra"):
        panel = repo_root / "results" / "predictions" / f"comparison_{species}.jpg"
        assert panel.exists() and panel.stat().st_size > 10_000

    readme = repo_root / "README.md"
    notebook = repo_root / "notebooks" / "treinando_yolo_african_wildlife.ipynb"
    readme_txt = readme.read_text(encoding="utf-8")
    assert readme.exists() and 600 < readme.stat().st_size < 1900
    assert "colab.research.google.com" not in readme_txt
    assert "p.p" not in readme_txt.lower()
    assert "porcentua" in readme_txt.lower()
    assert notebook.exists() and notebook.stat().st_size > 1_000


def test_bare_pytest_cli_and_colab_dataset_path_fallback(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    pytest_exe = Path(sys.executable).with_name("pytest.exe")
    proc = subprocess.run(
        [str(pytest_exe), "--collect-only", "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bare pytest failed: {proc.stdout}\n{proc.stderr}"

    nb_file = repo_root / "notebooks" / "treinando_yolo_african_wildlife.ipynb"
    nb_json = json.loads(nb_file.read_text(encoding="utf-8"))
    code_blob = "\n".join(
        "".join(c["source"]) for c in nb_json["cells"] if c["cell_type"] == "code"
    )
    assert "check_det_dataset" in code_blob
