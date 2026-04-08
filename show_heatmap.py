import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ergebnisse von AnomalyDINO
results_root = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\16-shot_preprocess=force_no_mask_no_rotation_bestsearch16_maxmap_norot\anomaly_maps\seed=0\buttons\test"
)

for npy_file in results_root.rglob("*.npy"):
    if "_png" in npy_file.parts:
        continue

    out_dir = npy_file.parent / "_png"
    out_dir.mkdir(exist_ok=True)

    rel_parent = npy_file.parent.relative_to(results_root)
    print(f"Verarbeite: {rel_parent}")

    arr = np.load(npy_file)

    plt.figure(figsize=(6, 4))
    plt.imshow(arr)
    plt.colorbar()
    plt.title(f"{rel_parent} | {npy_file.stem}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{npy_file.stem}.png", dpi=150)
    plt.close()

print("Fertig.")
