import os
import shutil
from config.config import PROCESSED_DATA_DIR, ANNOTATIONS_DIR, CLASSES_FILE, DATASET_SPLIT
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def run_training_pipeline():
    # ✅ Télécharger ou charger le modèle YOLOv8s pré-entraîné
    model = YOLO('yolov8s.pt')

    # Dossiers d'origine
    img_dir = PROCESSED_DATA_DIR
    label_dir = ANNOTATIONS_DIR

    # Dossiers cibles
    os.makedirs(os.path.join(DATASET_SPLIT, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_SPLIT, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_SPLIT, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_SPLIT, "labels/val"), exist_ok=True)

    # Liste des fichiers images ayant des annotations correspondantes
    images = [
        f for f in os.listdir(img_dir)
        if f.endswith('.tif') and os.path.exists(os.path.join(label_dir, f.replace(".tif", ".txt")))
    ]

    # Split train/val
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Copie des fichiers d'entraînement
    for img in train_imgs:
        label = img.replace(".tif", ".txt")
        shutil.copy(os.path.join(img_dir, img), os.path.join(DATASET_SPLIT, "images/train", img))
        shutil.copy(os.path.join(label_dir, label), os.path.join(DATASET_SPLIT, "labels/train", label))

    # Copie des fichiers de validation
    for img in val_imgs:
        label = img.replace(".tif", ".txt")
        shutil.copy(os.path.join(img_dir, img), os.path.join(DATASET_SPLIT, "images/val", img))
        shutil.copy(os.path.join(label_dir, label), os.path.join(DATASET_SPLIT, "labels/val", label))

    print("✅ Séparation terminée :")
    print(f"- Train : {len(train_imgs)} images")
    print(f"- Val   : {len(val_imgs)} images")
    print(f"Les données sont stockées dans {DATASET_SPLIT}")

    # ✅ Définir le chemin vers votre fichier data.yaml
    data_yaml_path = 'resources/config.yaml'

    # ✅ Lancer l'entraînement du modèle YOLOv8s sur vos données
    model.train(
        data=data_yaml_path,
        epochs=150,
        imgsz=640,
        batch=8,  # à ajuster selon ta VRAM
        name='yolov8s_all_classes',
        patience=20,
        project='runs/train'
    )
    # === Sauvegarde du modèle fine-tuné
    best_model_path = os.path.join('runs/train/yolov8s_all_classes/weights/best.pt')
    final_model_dir = os.path.join('resources', 'models', 'yolov8_fine_tun')
    os.makedirs(final_model_dir, exist_ok=True)
    shutil.copy(best_model_path, os.path.join(final_model_dir, 'YOLOv8-s_FT.pt'))

if __name__ == "__main__":
    run_training_pipeline()
