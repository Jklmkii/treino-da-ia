from __future__ import annotations

import random
import shutil
from pathlib import Path
import yaml

CLASS_NAMES: dict[int, str] = {
    0: "buffalo",
    1: "elephant",
    2: "rhino",
    3: "zebra",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_label_classes(label_path: Path) -> set[int]:
    """Retorna o conjunto de IDs de classes presentes em um arquivo de label YOLO."""
    if not label_path.exists():
        return set()
    classes: set[int] = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def _select_balanced_split(
    images_dir: Path,
    labels_dir: Path,
    per_class: int,
    rng: random.Random,
    globally_used_stems: set[str],
) -> list[tuple[Path, Path]]:
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    by_class: dict[int, list[tuple[Path, Path]]] = {cls_id: [] for cls_id in CLASS_NAMES}

    for img_path in all_images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        present_classes = parse_label_classes(lbl_path)
        for cls_id in present_classes:
            if cls_id in by_class:
                by_class[cls_id].append((img_path, lbl_path))

    for cls_id in by_class:
        rng.shuffle(by_class[cls_id])

    selected: list[tuple[Path, Path]] = []
    target_total = per_class * len(CLASS_NAMES)

    for cls_id in sorted(CLASS_NAMES):
        picked_for_cls = 0
        for img_path, lbl_path in by_class[cls_id]:
            if picked_for_cls >= per_class:
                break
            if img_path.stem in globally_used_stems:
                continue
            globally_used_stems.add(img_path.stem)
            selected.append((img_path, lbl_path))
            picked_for_cls += 1

    # Caso imagens multiclasse tenham reduzido o pool exclusivo de alguma classe, completa até target_total
    if len(selected) < target_total:
        remaining = [
            (img_path, labels_dir / f"{img_path.stem}.txt")
            for img_path in all_images
            if img_path.stem not in globally_used_stems
            and parse_label_classes(labels_dir / f"{img_path.stem}.txt")
        ]
        rng.shuffle(remaining)
        for pair in remaining:
            if len(selected) >= target_total:
                break
            globally_used_stems.add(pair[0].stem)
            selected.append(pair)

    return selected


def create_stratified_subset(
    source_root: Path,
    output_root: Path,
    train_per_class: int = 50,
    val_per_class: int = 15,
    seed: int = 42,
) -> Path:
    """Cria um subconjunto estratificado e sem vazamento entre train e val no formato YOLO."""
    rng = random.Random(seed)
    if output_root.exists():
        shutil.rmtree(output_root)

    globally_used_stems: set[str] = set()
    splits_cfg = [
        ("train", train_per_class),
        ("val", val_per_class),
    ]

    for split_name, per_class in splits_cfg:
        src_imgs = source_root / "images" / split_name
        src_lbls = source_root / "labels" / split_name
        dst_imgs = output_root / "images" / split_name
        dst_lbls = output_root / "labels" / split_name
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        chosen = _select_balanced_split(
            images_dir=src_imgs,
            labels_dir=src_lbls,
            per_class=per_class,
            rng=rng,
            globally_used_stems=globally_used_stems,
        )
        for img_path, lbl_path in chosen:
            shutil.copy2(img_path, dst_imgs / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbls / lbl_path.name)
            else:
                (dst_lbls / f"{img_path.stem}.txt").write_text("", encoding="utf-8")

    yaml_path = output_root / "african_wildlife_subset.yaml"
    yaml_content = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    yaml_path.write_text(
        yaml.safe_dump(yaml_content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path.resolve()


def prepare_african_wildlife_subset(workspace_dir: Path) -> Path:
    """Baixa oficialmente a base african-wildlife via Ultralytics e gera o recorte estratificado."""
    from ultralytics import settings
    from ultralytics.data.utils import check_det_dataset

    datasets_dir = (workspace_dir / "datasets").resolve()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"datasets_dir": str(datasets_dir)})

    dataset_info = check_det_dataset("african-wildlife.yaml")
    source_root = Path(dataset_info["path"])
    subset_root = datasets_dir / "african_wildlife_subset"

    yaml_path = create_stratified_subset(
        source_root=source_root,
        output_root=subset_root,
        train_per_class=50,
        val_per_class=15,
        seed=42,
    )
    return yaml_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_yaml = prepare_african_wildlife_subset(root)
    print(f"Subset YAML pronto em: {out_yaml}")
