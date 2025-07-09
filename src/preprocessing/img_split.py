import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config.config import RAW_DATA_DIR, TILE_SIZE, PROCESSED_DATA_DIR, COORDS_DIR


def tile_image_with_coords(image_path, tile_size=640, output_dir="tiles", coords_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dossier pour les coordonnées (optionnel)
    if coords_dir is None:
        coords_dir = output_dir / "coords"
    else:
        coords_dir = Path(coords_dir)
    coords_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        count = src.count  # Nombre de bandes dans l'image

        stride = tile_size // 2  # 50% de recouvrement

        print(f"Image size: {width}x{height} | Stride: {stride} | Tile size: {tile_size}")
        print(f"Nombre de bandes : {count}")

        for y in tqdm(range(0, height - tile_size + 1, stride), desc="Découpage vertical"):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)

                # Lire les 3 bandes RGB
                rgb = src.read([1, 2, 3], window=window)

                # Lire la 4e bande (coordonnées), si disponible
                if count >= 4:
                    coords_band = src.read(4, window=window)

                # Enregistrer le patch RGB
                tile_id = f"{y}_{x}"
                rgb_path = output_dir / f"tile_{tile_id}.tif"

                profile = src.profile.copy()
                profile.update({
                    "count": 3,
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src.transform),
                    "driver": "GTiff"
                })

                with rasterio.open(rgb_path, 'w', **profile) as dst:
                    dst.write(rgb)

                # Enregistrer les coordonnées si présentes
                if count >= 4:
                    np.save(coords_dir / f"tile_{tile_id}.npy", coords_band)

    print(f"\n✅ Tuiles RGB enregistrées dans : {output_dir}")
    print(f"✅ Coordonnées enregistrées dans : {coords_dir}")


# ========== Utilisation ==========

if __name__ == "__main__":
    tile_image_with_coords(
        image_path=RAW_DATA_DIR + "img.tif",
        tile_size=TILE_SIZE,
        output_dir=PROCESSED_DATA_DIR,
        coords_dir=COORDS_DIR
    )
