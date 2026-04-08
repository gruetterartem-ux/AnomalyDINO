import numpy as np
import cv2
from pathlib import Path

# Roh-Heatmaps von AnomalyDINO
results_root = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vits14_448\1-shot_preprocess=agnostic\anomaly_maps\seed=0\buttons-fusion\test"
)

# Originalbilder
images_root = Path(r"C:\anomalydino_data\buttons\test")

for class_dir in results_root.iterdir():
    if not class_dir.is_dir():
        continue

    image_class_dir = images_root / class_dir.name
    if not image_class_dir.exists():
        print(f"Kein passender Bildordner gefunden für {class_dir.name}")
        continue

    out_dir = class_dir / "_overlay"
    out_dir.mkdir(exist_ok=True)

    print(f"Verarbeite: {class_dir.name}")

    for npy_file in class_dir.glob("*.npy"):
        stem = npy_file.stem

        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]:
            candidate = image_class_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"Kein Originalbild gefunden für {class_dir.name}/{stem}")
            continue

        heatmap = np.load(npy_file)
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Heatmap auf Originalgröße bringen
        heatmap_up = cv2.resize(
            heatmap,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        # auf 0..255 normieren
        hm_norm = heatmap_up - heatmap_up.min()
        if hm_norm.max() > 0:
            hm_norm = hm_norm / hm_norm.max()
        hm_uint8 = (hm_norm * 255).astype(np.uint8)

        # farbige Heatmap
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        # Overlay
        alpha = 0.45
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, hm_color, alpha, 0)

        out_path = out_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Gespeichert:", out_path)

print("Fertig.")