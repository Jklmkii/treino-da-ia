from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from src.dataset_prep import CLASS_NAMES, parse_label_classes, prepare_african_wildlife_subset

SPECIES_PT: dict[str, str] = {
    "buffalo": "Búfalo (buffalo)",
    "elephant": "Elefante (elephant)",
    "rhino": "Rinoceronte (rhino)",
    "zebra": "Zebra (zebra)",
}

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (230, 126, 34),   # buffalo - laranja
    1: (46, 204, 113),   # elephant - verde
    2: (155, 89, 182),   # rhino - roxo
    3: (52, 152, 219),   # zebra - azul
}


def _load_font(size: int = 18) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Users/lucas/AppData/Roaming/Ultralytics/Arial.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_ground_truth_boxes(image_path: Path, label_path: Path) -> Image.Image:
    """Desenha as bounding boxes reais (Ground Truth) na imagem original."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(size=max(14, img.width // 38))
    w, h = img.size

    if label_path.exists():
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            color = CLASS_COLORS_RGB.get(cls_id, (255, 215, 0))
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            caption = f"GT: {cls_name}"
            bbox = draw.textbbox((x1, max(0, y1 - 22)), caption, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, max(0, y1 - 22)), caption, fill=(255, 255, 255), font=font)

    return img


def build_side_by_side_panel(
    images: list[tuple[str, Image.Image]],
    output_path: Path,
    title: str,
    target_height: int = 380,
) -> Path:
    """Combina múltiplas imagens lado a lado com cabeçalho e legendas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized: list[tuple[str, Image.Image]] = []
    for label, im in images:
        rgb = im.convert("RGB")
        ratio = target_height / rgb.height
        new_w = max(1, int(rgb.width * ratio))
        resized.append((label, rgb.resize((new_w, target_height), Image.Resampling.LANCZOS)))

    padding = 16
    header_h = 54
    sub_h = 34
    total_w = sum(im.width for _, im in resized) + padding * (len(resized) + 1)
    total_h = header_h + sub_h + target_height + padding

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22)
    sub_font = _load_font(17)

    draw.text((padding, 14), title, fill=(245, 247, 250), font=title_font)

    x_cursor = padding
    for label, im in resized:
        draw.text((x_cursor + 6, header_h + 4), label, fill=(144, 202, 249), font=sub_font)
        canvas.paste(im, (x_cursor, header_h + sub_h))
        x_cursor += im.width + padding

    canvas.save(output_path, quality=93)
    return output_path


def build_2x2_showcase(images: list[tuple[str, Image.Image]], output_path: Path, header: str) -> Path:
    """Cria um mosaico 2x2 destacando as 4 espécies com bounding boxes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 520, 360
    pad = 16
    top_h = 56
    canvas = Image.new("RGB", (cell_w * 2 + pad * 3, top_h + (cell_h + 30) * 2 + pad * 2), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22)
    sub_font = _load_font(16)
    draw.text((pad, 15), header, fill=(255, 255, 255), font=title_font)

    for idx, (caption, im) in enumerate(images[:4]):
        row, col = divmod(idx, 2)
        x = pad + col * (cell_w + pad)
        y = top_h + row * (cell_h + 30 + pad)
        fitted = im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        draw.text((x + 4, y), caption, fill=(129, 212, 250), font=sub_font)
        canvas.paste(fitted, (x, y + 26))

    canvas.save(output_path, quality=93)
    return output_path


def generate_comparison_charts(
    summary: dict[str, dict[str, Any]],
    charts_dir: Path,
    histories: dict[str, list[dict[str, float]]] | None = None,
) -> list[Path]:
    """Gera gráficos comparativos de precisão/mAP, eficiência e curvas de aprendizado."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    models = list(summary.keys())

    # 1. Gráfico de barras de qualidade de detecção
    metric_keys = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP@50"),
        ("map50_95", "mAP@50-95"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    x_positions = list(range(len(metric_keys)))
    bar_w = 0.34
    palette = ["#2563eb", "#10b981", "#f59e0b"]

    for m_idx, model_name in enumerate(models):
        offset = (m_idx - (len(models) - 1) / 2) * bar_w
        vals = [float(summary[model_name][k]) for k, _ in metric_keys]
        bars = ax.bar(
            [x + offset for x in x_positions],
            vals,
            width=bar_w,
            label=model_name,
            color=palette[m_idx % len(palette)],
            edgecolor="#1e293b",
            linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Pontuação (0.0 – 1.0)", fontsize=11)
    ax.set_title("Comparativo de Desempenho: YOLOv8n vs YOLO11n (African Wildlife)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=10.5)
    fig.tight_layout()
    metrics_png = charts_dir / "metrics_comparison.png"
    fig.savefig(metrics_png)
    plt.close(fig)
    saved_paths.append(metrics_png)

    # 2. Gráfico de eficiência (Inferência ms, Tempo de Treino s, Parâmetros M)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), dpi=180)
    eff_specs = [
        ("inference_ms", "Inferência (ms / img)", "Menor é mais rápido"),
        ("train_time_s", "Tempo de Treino (s)", "15 épocas em CPU"),
        ("params_m", "Parâmetros (Milhões)", "Leveza da arquitetura"),
    ]
    for ax_i, (key, title, subtitle) in zip(axes, eff_specs):
        vals = [float(summary[m][key]) for m in models]
        bars = ax_i.bar(models, vals, color=palette[: len(models)], width=0.52, edgecolor="#1e293b")
        max_v = max(vals) if vals else 1.0
        ax_i.set_ylim(0, max_v * 1.24 if max_v > 0 else 1.0)
        ax_i.set_title(f"{title}\n({subtitle})", fontsize=10.5, fontweight="bold")
        ax_i.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v in zip(bars, vals):
            ax_i.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.03,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.suptitle("Eficiência Computacional: YOLOv8n vs YOLO11n", fontsize=13, fontweight="bold")
    fig.tight_layout()
    eff_png = charts_dir / "efficiency_comparison.png"
    fig.savefig(eff_png)
    plt.close(fig)
    saved_paths.append(eff_png)

    # 3. Curvas de aprendizado por época (se disponíveis)
    if histories:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
        for m_idx, (model_name, rows) in enumerate(histories.items()):
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            map50_curve = [r["map50"] for r in rows]
            box_loss_curve = [r["box_loss"] for r in rows]
            color = palette[m_idx % len(palette)]
            ax1.plot(epochs, map50_curve, marker="o", linewidth=2, label=model_name, color=color)
            ax2.plot(epochs, box_loss_curve, marker="s", linewidth=2, label=model_name, color=color)

        ax1.set_title("Evolução do mAP@50 por Época", fontsize=11.5, fontweight="bold")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("mAP@50")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

        ax2.set_title("Queda da Box Loss no Treino", fontsize=11.5, fontweight="bold")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("train/box_loss")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend()

        fig.suptitle("Curvas de Treinamento: YOLOv8n vs YOLO11n", fontsize=13, fontweight="bold")
        fig.tight_layout()
        curves_png = charts_dir / "training_curves.png"
        fig.savefig(curves_png)
        plt.close(fig)
        saved_paths.append(curves_png)

    return saved_paths


def _parse_ultralytics_history(results_csv: Path) -> list[dict[str, float]]:
    if not results_csv.exists():
        return []
    parsed: list[dict[str, float]] = []
    with results_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}
            try:
                parsed.append(
                    {
                        "epoch": float(clean.get("epoch", 0)),
                        "map50": float(clean.get("metrics/mAP50(B)", 0.0)),
                        "map50_95": float(clean.get("metrics/mAP50-95(B)", 0.0)),
                        "box_loss": float(clean.get("train/box_loss", 0.0)),
                    }
                )
            except ValueError:
                continue
    return parsed


def run_benchmark(
    yaml_path: Path,
    workspace_dir: Path,
    epochs: int = 15,
    imgsz: int = 416,
    batch: int = 16,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Treina e valida YOLOv8n e YOLO11n sob condições idênticas e salva métricas e gráficos."""
    from ultralytics import YOLO

    runs_dir = (workspace_dir / "runs" / "detect").resolve()
    results_dir = (workspace_dir / "results").resolve()
    charts_dir = results_dir / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("YOLOv8n", "yolov8n.pt", "yolov8n_wildlife"),
        ("YOLO11n", "yolo11n.pt", "yolo11n_wildlife"),
    ]

    summary: dict[str, dict[str, Any]] = {}
    best_weights: dict[str, Path] = {}
    histories: dict[str, list[dict[str, float]]] = {}

    for display_name, base_weights, run_name in candidates:
        print(f"\n=== Iniciando treinamento de {display_name} ({base_weights}) ===")
        init_model = YOLO(base_weights)
        params_m = sum(p.numel() for p in init_model.model.parameters()) / 1e6

        t0 = time.perf_counter()
        init_model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            workers=0,
            seed=42,
            deterministic=True,
            amp=False,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
        )
        train_elapsed = time.perf_counter() - t0

        run_folder = runs_dir / run_name
        best_pt = run_folder / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = run_folder / "weights" / "last.pt"
        best_weights[display_name] = best_pt

        trained_model = YOLO(str(best_pt))
        val_metrics = trained_model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            workers=0,
            plots=False,
        )

        per_class_ap50: dict[str, float] = {}
        ap_indices = list(getattr(val_metrics, "ap_class_index", []))
        ap50_values = list(getattr(val_metrics.box, "ap50", []))
        for cls_idx, ap_val in zip(ap_indices, ap50_values):
            cls_label = CLASS_NAMES.get(int(cls_idx), str(cls_idx))
            per_class_ap50[cls_label] = round(float(ap_val), 4)

        summary[display_name] = {
            "weights": base_weights,
            "precision": round(float(val_metrics.box.mp), 4),
            "recall": round(float(val_metrics.box.mr), 4),
            "map50": round(float(val_metrics.box.map50), 4),
            "map50_95": round(float(val_metrics.box.map), 4),
            "inference_ms": round(float(val_metrics.speed.get("inference", 0.0)), 2),
            "train_time_s": round(float(train_elapsed), 1),
            "params_m": round(float(params_m), 2),
            "per_class_ap50": per_class_ap50,
        }
        histories[display_name] = _parse_ultralytics_history(run_folder / "results.csv")

    # Salva JSON e CSV consolidados
    json_path = results_dir / "metrics_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = results_dir / "metrics_comparison.csv"
    fieldnames = [
        "model",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "inference_ms",
        "train_time_s",
        "params_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, data in summary.items():
            row = {"model": model_name}
            row.update({k: data[k] for k in fieldnames if k != "model"})
            writer.writerow(row)

    generate_comparison_charts(summary, charts_dir, histories=histories)
    return summary, best_weights


def generate_prediction_panels(
    best_weights: dict[str, Path],
    val_images_dir: Path,
    val_labels_dir: Path,
    preds_dir: Path,
    imgsz: int = 416,
) -> list[Path]:
    """Gera imagens com marcações de bounding boxes e pranchas comparativas para as 4 espécies."""
    from ultralytics import YOLO

    preds_dir.mkdir(parents=True, exist_ok=True)
    loaded_models = {name: YOLO(str(pt)) for name, pt in best_weights.items()}

    all_val_images = sorted(
        p for p in val_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    saved_files: list[Path] = []
    showcase_v8: list[tuple[str, Image.Image]] = []
    showcase_v11: list[tuple[str, Image.Image]] = []

    for cls_id, cls_name in sorted(CLASS_NAMES.items()):
        candidates = [
            img_p for img_p in all_val_images
            if cls_id in parse_label_classes(val_labels_dir / f"{img_p.stem}.txt")
        ]
        if not candidates:
            continue

        # Escolhe a imagem onde a soma das confianças da classe alvo nos dois modelos é maior
        best_img = candidates[0]
        best_score = -1.0
        best_renders: dict[str, Image.Image] = {}

        for cand in candidates[:10]:
            score = 0.0
            cand_renders: dict[str, Image.Image] = {}
            for m_name, model in loaded_models.items():
                res = model.predict(source=str(cand), imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
                bgr = res.plot(line_width=2, font_size=12)
                cand_renders[m_name] = Image.fromarray(bgr[:, :, ::-1])
                if res.boxes is not None and len(res.boxes) > 0:
                    confs = res.boxes.conf.cpu().tolist()
                    clss = [int(c) for c in res.boxes.cls.cpu().tolist()]
                    for c_val, cf in zip(clss, confs):
                        if c_val == cls_id:
                            score += cf
            if score > best_score:
                best_score = score
                best_img = cand
                best_renders = cand_renders

        gt_img = draw_ground_truth_boxes(best_img, val_labels_dir / f"{best_img.stem}.txt")

        # Salva predições individuais com bounding boxes
        for m_name, rendered_im in best_renders.items():
            slug = m_name.lower()
            indiv_path = preds_dir / f"{slug}_{cls_name}.jpg"
            rendered_im.save(indiv_path, quality=93)
            saved_files.append(indiv_path)

        if "YOLOv8n" in best_renders:
            showcase_v8.append((SPECIES_PT[cls_name], best_renders["YOLOv8n"]))
        if "YOLO11n" in best_renders:
            showcase_v11.append((SPECIES_PT[cls_name], best_renders["YOLO11n"]))

        panel_items = [("Ground Truth (Real)", gt_img)]
        for m_name in ["YOLOv8n", "YOLO11n"]:
            if m_name in best_renders:
                panel_items.append((f"{m_name} (Predição)", best_renders[m_name]))

        panel_path = preds_dir / f"comparison_{cls_name}.jpg"
        build_side_by_side_panel(
            images=panel_items,
            output_path=panel_path,
            title=f"Detecção com Bounding Boxes — {SPECIES_PT[cls_name]}",
        )
        saved_files.append(panel_path)

    if len(showcase_v8) == 4:
        saved_files.append(
            build_2x2_showcase(
                showcase_v8,
                preds_dir / "yolov8n_showcase_grid.jpg",
                "YOLOv8n — Resultados com Marcações de Bounding Boxes (4 Espécies)",
            )
        )
    if len(showcase_v11) == 4:
        saved_files.append(
            build_2x2_showcase(
                showcase_v11,
                preds_dir / "yolo11n_showcase_grid.jpg",
                "YOLO11n — Resultados com Marcações de Bounding Boxes (4 Espécies)",
            )
        )

    return saved_files


def main() -> None:
    workspace_dir = Path(__file__).resolve().parents[1]
    subset_yaml = workspace_dir / "datasets" / "african_wildlife_subset" / "african_wildlife_subset.yaml"
    if not subset_yaml.exists():
        subset_yaml = prepare_african_wildlife_subset(workspace_dir)

    summary, best_weights = run_benchmark(
        yaml_path=subset_yaml,
        workspace_dir=workspace_dir,
        epochs=15,
        imgsz=416,
        batch=16,
    )

    subset_root = subset_yaml.parent
    preds = generate_prediction_panels(
        best_weights=best_weights,
        val_images_dir=subset_root / "images" / "val",
        val_labels_dir=subset_root / "labels" / "val",
        preds_dir=workspace_dir / "results" / "predictions",
        imgsz=416,
    )
    print("\n=== Resumo Final ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Total de imagens/painéis com bounding boxes gerados: {len(preds)}")


if __name__ == "__main__":
    main()
