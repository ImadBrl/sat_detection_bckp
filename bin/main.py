import gradio as gr
import shutil
import os
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from src.preprocessing.img_split import tile_image_with_coords
from notebooks.generate_report import generate_report

# === CONFIGURATION ===
YOLO_MODEL_PATH = "resources/models/YOLOv8-s_FT.pt"
CLASSES_FILE = "resources/classes.txt"
TILE_SIZE = 640


def create_session_folders():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:5]
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


def full_pipeline(image_file):
    if not image_file:
        raise ValueError("❌ Aucune image satellite fournie.")

    # === Créer structure de session ===
    folders = create_session_folders()
    input_path = folders["input"] / "input_image.tif"
    shutil.copy(image_file.name, input_path)

    print("✅ Étape 1 : Découpage de l’image")
    tile_image_with_coords(
        image_path=str(input_path),
        tile_size=TILE_SIZE,
        output_dir=str(folders["tiles"]),
        coords_dir=folders["base"] / "coords"  # Pour garder l’organisation
    )

    print("✅ Étape 2 : Chargement du modèle YOLOv8 FT")
    model = YOLO(YOLO_MODEL_PATH)

    print("✅ Étape 3 : Prédiction sur les tuiles")
    tile_files = list(Path(folders["tiles"]).glob("*.tif"))
    for tile_path in tile_files:
        model.predict(
            source=str(tile_path),
            save=False,
            save_txt=True,
            project=str(folders["annotations"]),
            name=".",  # Éviter les sous-dossiers
            exist_ok=True
        )

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
    fn=full_pipeline,
    inputs=gr.File(file_types=[".tif"], label="🛰️ Charger une image satellite (.tif)"),
    outputs=gr.File(label="📄 Télécharger le rapport généré"),
    title="Détection d’objets sur images satellites",
    description="Ce système découpe automatiquement une image satellite, applique YOLOv8 pour détecter les objets, puis génère un rapport.",
)

if __name__ == "__main__":
    iface.launch()
