import os
from collections import defaultdict
from pathlib import Path

def generate_report(annotations_dir, classes_file, output_file):
    annotations_dir = Path(annotations_dir)

    # Aller chercher tous les .txt dans les sous-dossiers (ex: annotations/labels/)
    annotation_files = list(annotations_dir.rglob("*.txt"))
    annotation_files = [f for f in annotation_files if f.name != "classes.txt"]

    if not annotation_files:
        print(f"⚠️ Aucun fichier .txt trouvé dans {annotations_dir}")
        return

    # Charger les classes
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    descriptions = []

    for file_path in annotation_files:
        counts = defaultdict(int)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_name = classes[class_id]
                    counts[class_name] += 1

        # Extraire les coordonnées depuis le nom du fichier
        try:
            y, x = file_path.stem.replace("tile_", "").split("_")
        except ValueError:
            y, x = "?", "?"

        desc = f"Tuiles ({x}px, {y}px) : "
        if not counts:
            desc += "aucun objet détecté."
        else:
            desc += ", ".join(f"{v} {k}(s)" for k, v in counts.items()) + "."

        descriptions.append(desc)

    # Créer le dossier du rapport s’il n’existe pas
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(descriptions))

    print(f"✅ Rapport généré dans : {output_file}")
