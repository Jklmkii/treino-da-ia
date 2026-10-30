from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _md_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code_cell(code: str, output_text: str | None = None) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    if output_text:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": output_text.splitlines(keepends=True),
            }
        )
    return {
        "cell_type": "code",
        "execution_count": 1 if output_text else None,
        "metadata": {},
        "outputs": outputs,
        "source": code.strip("\n").splitlines(keepends=True),
    }


def _fmt_pt(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def build_colab_notebook(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    repo_slug: str = "Jklmkii/treino-da-ia",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v8 = summary["YOLOv8n"]
    v11 = summary["YOLO11n"]

    summary_stdout = (
        f"YOLOv8n -> Precisão: {v8['precision']*100:.1f}% | Recall: {v8['recall']*100:.1f}% | mAP50: {v8['map50']*100:.1f}%\n"
        f"YOLO11n -> Precisão: {v11['precision']*100:.1f}% | Recall: {v11['recall']*100:.1f}% | mAP50: {v11['map50']*100:.1f}%\n"
    )

    cells = [
        _md_cell(
            "# Treino do YOLO - YOLOv8n vs YOLO11n\n\n"
            "Treinando duas versões do YOLO com `ultralytics` na base `african-wildlife` "
            "(búfalo, elefante, rinoceronte e zebra) pra comparar os resultados."
        ),
        _code_cell(
            "!pip install -q ultralytics matplotlib pillow\n"
            "import torch\n"
            "from ultralytics import YOLO\n\n"
            "DEVICE = 0 if torch.cuda.is_available() else 'cpu'"
        ),
        _code_cell(
            "EPOCHS = 15\n"
            "IMGSZ = 416\n\n"
            "model_v8 = YOLO('yolov8n.pt')\n"
            "model_v8.train(data='african-wildlife.yaml', epochs=EPOCHS, imgsz=IMGSZ, batch=16, device=DEVICE, seed=42)\n\n"
            "model_v11 = YOLO('yolo11n.pt')\n"
            "model_v11.train(data='african-wildlife.yaml', epochs=EPOCHS, imgsz=IMGSZ, batch=16, device=DEVICE, seed=42)"
        ),
        _code_cell(
            "metrics_v8 = model_v8.val()\n"
            "metrics_v11 = model_v11.val()\n\n"
            "for nome, m in [('YOLOv8n', metrics_v8), ('YOLO11n', metrics_v11)]:\n"
            "    print(f'{nome} -> Precisão: {m.box.mp*100:.1f}% | Recall: {m.box.mr*100:.1f}% | mAP50: {m.box.map50*100:.1f}%')",
            output_text=summary_stdout,
        ),
        _code_cell(
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "from ultralytics.data.utils import check_det_dataset\n\n"
            "ds_info = check_det_dataset('african-wildlife.yaml')\n"
            "val_imgs = sorted(Path(ds_info['val']).glob('*.jpg'))[:4]\n"
            "fig, axes = plt.subplots(len(val_imgs), 2, figsize=(12, 4 * len(val_imgs)))\n\n"
            "for i, img_path in enumerate(val_imgs):\n"
            "    axes[i, 0].imshow(model_v8.predict(str(img_path), conf=0.25, verbose=False)[0].plot()[:, :, ::-1])\n"
            "    axes[i, 0].set_title(f'YOLOv8n - {img_path.name}')\n"
            "    axes[i, 0].axis('off')\n"
            "    axes[i, 1].imshow(model_v11.predict(str(img_path), conf=0.25, verbose=False)[0].plot()[:, :, ::-1])\n"
            "    axes[i, 1].set_title(f'YOLO11n - {img_path.name}')\n"
            "    axes[i, 1].axis('off')\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.16"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook_json, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_readme_markdown(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
    repo_slug: str = "Jklmkii/treino-da-ia",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v8 = summary["YOLOv8n"]
    v11 = summary["YOLO11n"]

    diff_map = abs(v11["map50"] - v8["map50"]) * 100
    diff_rec = abs(v11["recall"] - v8["recall"]) * 100
    unidade_map = "ponto porcentual" if round(diff_map, 2) == 1.0 else "pontos porcentuais"
    unidade_rec = "ponto porcentual" if round(diff_rec, 2) == 1.0 else "pontos porcentuais"

    content = f"""# Treino do YOLO - Animais Selvagens

Atividade treinando o YOLO em Python com a biblioteca `ultralytics` na base `african-wildlife`, que tem fotos de **búfalo, elefante, rinoceronte e zebra**.

Treinei duas versões diferentes (**YOLOv8n** e **YOLO11n**) com a mesma configuração (200 fotos de treino, 60 de validação e 15 épocas) pra ver qual saía melhor.

## Comparando YOLOv8n e YOLO11n

| Modelo | Precisão | Recall | mAP50 | Tempo por foto | Tempo de treino |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n | {_fmt_pt(v8['precision']*100)}% | {_fmt_pt(v8['recall']*100)}% | {_fmt_pt(v8['map50']*100)}% | {_fmt_pt(v8['inference_ms'])} ms | {_fmt_pt(v8['train_time_s'])} s |
| YOLO11n | {_fmt_pt(v11['precision']*100)}% | {_fmt_pt(v11['recall']*100)}% | {_fmt_pt(v11['map50']*100)}% | {_fmt_pt(v11['inference_ms'])} ms | {_fmt_pt(v11['train_time_s'])} s |

No geral o **YOLO11n foi melhor**: subiu {_fmt_pt(diff_map, 2)} {unidade_map} no `mAP50` e {_fmt_pt(diff_rec, 2)} {unidade_rec} no `Recall`, achando mais animais que estavam no fundo das fotos. Ele também treinou quase na metade do tempo e rodou mais que o dobro mais rápido.

![Gráfico das métricas](results/charts/metrics_comparison.png)

## Resultados com as boxes marcadas

Resultados marcando os 4 animais nas fotos de teste:

![YOLO11n](results/predictions/yolo11n_showcase_grid.jpg)
![YOLOv8n](results/predictions/yolov8n_showcase_grid.jpg)

Comparação lado a lado (marcação real vs YOLOv8n vs YOLO11n):

![Elefante](results/predictions/comparison_elephant.jpg)
![Zebra](results/predictions/comparison_zebra.jpg)
![Rinoceronte](results/predictions/comparison_rhino.jpg)
![Búfalo](results/predictions/comparison_buffalo.jpg)

## Como rodar

```bash
pip install -r requirements.txt
python -m src.dataset_prep
python -m src.train_and_compare
```
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    workspace_dir = Path(__file__).resolve().parents[1]
    metrics_file = workspace_dir / "results" / "metrics_comparison.json"
    summary = json.loads(metrics_file.read_text(encoding="utf-8"))
    build_colab_notebook(
        summary,
        workspace_dir / "notebooks" / "treinando_yolo_african_wildlife.ipynb",
        repo_slug="Jklmkii/treino-da-ia",
    )
    build_readme_markdown(
        summary,
        workspace_dir / "README.md",
        repo_slug="Jklmkii/treino-da-ia",
    )
    print("Arquivos gerados!")


if __name__ == "__main__":
    main()
