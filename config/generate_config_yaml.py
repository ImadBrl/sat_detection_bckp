from config import CLASSES_FILE, DATASET_SPLIT, YAML_FILE
import yaml

# Lire les classes depuis le fichier
with open(CLASSES_FILE, 'r') as f:
    class_names = [line.strip() for line in f if line.strip()]

# Générer le dictionnaire YAML
data_yaml = {
    'path': DATASET_SPLIT,         # Chemin vers le dossier contenant images/ et labels/
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(class_names),
    'names': class_names
}

# Créer le dossier si besoin
import os
os.makedirs("config", exist_ok=True)

# Sauvegarder le fichier YAML
with open(YAML_FILE, 'w') as f:
    yaml.dump(data_yaml, f, sort_keys=False)


print("✅ Fichier config.yaml généré avec succès.")
