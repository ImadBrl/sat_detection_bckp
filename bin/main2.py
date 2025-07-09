import gradio as gr
import shutil
import os
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from src.preprocessing.img_split import tile_image_with_coords
from notebooks.generate_report import generate_report
from src.annotations.generate_annotations import generate_annotations
from notebooks.fine_tune_yolov8s import run_training_pipeline

# === CONFIGURATION ===
TILE_SIZE = 640
CLASSES_FILE = "resources/classes.txt"


def create_session_folders():
    session_id = datetime.now().strftime("%Y%m%d_%H:%M") + "_" + uuid4().hex[:5]
    base = Path("use") / session_id
    input_dir = base / "input"
    tiles_dir = base / "tiles"
    annotations_dir = base / "annotations"
    report_dir = base / "report"
    for d in [input_dir, tiles_dir, annotations_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "session_id": session_id,
        "base": base,
        "input": input_dir,
        "tiles": tiles_dir,
        "annotations": annotations_dir,
        "report": report_dir
    }


def pipeline_with_llm(image_file):
    if not image_file:
        raise ValueError("Veuillez fournir une image satellite .tif")

    # === Créer la session ===
    folders = create_session_folders()
    input_path = folders["input"] / "input_image.tif"
    shutil.copy(image_file.name, input_path)

    print("✅ Étape 1 : Découpage de l'image")
    tile_image_with_coords(
        image_path=str(input_path),
        tile_size=TILE_SIZE,
        output_dir=str(folders["tiles"])
    )

    print("✅ Étape 2 : Annotation avec Florence-2")
    generate_annotations(str(folders["tiles"]), str(folders["annotations"]))

    print("✅ Étape 3 : Fine-tuning de YOLOv8s sur les nouvelles annotations")
    run_training_pipeline()

    print("✅ Étape 4 : Génération du rapport")
    report_path = folders["report"] / "rapport.txt"

    generate_report(
        annotations_dir=str(folders["annotations"]),
        classes_file=CLASSES_FILE,
        output_file=str(report_path)
    )

    return str(report_path)


# === Interface Gradio ===
iface = gr.Interface(
    fn=pipeline_with_llm,
    inputs=gr.File(file_types=[".tif"], label="🛰️ Charger une image satellite (.tif)"),
    outputs=gr.File(label="📄 Rapport généré"),
    title="Annotation LLM + Fine-tuning YOLO + Rapport",
    description="Chargez une image satellite. Elle sera annotée avec Florence-2, utilisée pour fine-tuner YOLOv8, puis un rapport vous sera fourni."
)

if __name__ == "__main__":
    iface.launch()
