from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from .utils import get_dataset_info, dists2map, list_image_files, IMAGE_EXTENSIONS

from  matplotlib.colors import LinearSegmentedColormap
neon_violet = (0.5, 0.1, 0.5, 0.4)
neon_yellow = (0.8, 1.0, 0.02, 0.7)
red_gt = (1.0, 0, 0.0, 0.5)
colors = [(1.0, 1, 1.0, 0.0),  neon_violet, neon_yellow]
cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)


def get_test_gt_map(object_name, anomaly_type, img_nr, experiment, data_root, dataset = "MVTec", good=False):
    """
    Return test sample, ground truth (if not a good sample) and anomaly maps for given experiment and img_nr.
    """ 
    # test sample
    img_test_path = os.path.join(data_root, object_name, "test", anomaly_type, img_nr)
    image_test = cv2.cvtColor(cv2.imread(img_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    img_nr = os.path.splitext(img_nr)[0]

    # ground truth
    if not good:
        gt_path = os.path.join(data_root, object_name, "ground_truth", anomaly_type, img_nr)
        gt_path += "_mask.png" if dataset == "MVTec" else ".png"
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    
    # load patch distances for test sample
    dists = np.load(os.path.join(experiment, object_name, "test", anomaly_type, img_nr + ".npy"))
    
    # anomaly maps
    anomaly_map = dists2map(dists, image_test.shape)
    if good:
        return image_test, anomaly_map
    else:
      return image_test, gt_mask, anomaly_map


def plot_sample(image_test, anomaly_map, axs, cmap, vmax):
    vmax = max(float(vmax), float(np.nanmax(anomaly_map)), 0.0)
    axs.imshow(image_test)
    axs.imshow(anomaly_map, cmap=cmap, vmin=0.0, vmax=vmax)


def infer_vmax(exp_path, objects):
    vmax = {}
    for object_name in objects:
        current_max = 0
        test_root = os.path.join(exp_path, object_name, "test")
        for root, _, filenames in os.walk(test_root):
            for filename in filenames:
                if filename.endswith(".npy"):
                    max_score = np.load(os.path.join(root, filename)).max()
                    current_max = max(current_max, max_score)

        vmax[object_name] = current_max * 1.0
    return vmax


def _infer_objects_from_anomaly_maps_dir(anomaly_maps_dir):
    return sorted(
        [
            entry for entry in os.listdir(anomaly_maps_dir)
            if os.path.isdir(os.path.join(anomaly_maps_dir, entry))
        ]
    )


def _find_source_image(data_root, object_name, split_name, rel_path_no_ext):
    base_path = os.path.join(data_root, object_name, split_name, rel_path_no_ext)
    for ext in IMAGE_EXTENSIONS:
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find source image for {base_path} with known image extensions.")


def _save_patch_distance_grid_png(dists, output_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    color_plot = ax.imshow(dists)
    plt.colorbar(color_plot, ax=ax, fraction=0.12, pad=0.05, orientation="horizontal")
    ax.axis("off")
    ax.set_title("Patch Distances (1NN)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def export_anomaly_map_pngs(anomaly_maps_dir, data_root):
    objects = _infer_objects_from_anomaly_maps_dir(anomaly_maps_dir)
    if not objects:
        print(f"No object folders found in anomaly map directory: {anomaly_maps_dir}")
        return

    vmax = infer_vmax(anomaly_maps_dir, objects)
    exported_overlays = 0
    exported_patch_grids = 0

    for object_name in tqdm(objects, desc="Export anomaly map PNGs"):
        object_root = os.path.join(anomaly_maps_dir, object_name)
        for root, _, filenames in os.walk(object_root):
            for filename in filenames:
                if not filename.endswith(".npy"):
                    continue

                npy_path = os.path.join(root, filename)
                rel_path = os.path.relpath(npy_path, anomaly_maps_dir).replace("\\", "/")
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                rel_parts = rel_path_no_ext.split("/")
                if len(rel_parts) < 3:
                    continue

                split_name = rel_parts[1]
                source_rel_no_ext = os.path.join(*rel_parts[2:])
                image_path = _find_source_image(data_root, object_name, split_name, source_rel_no_ext)
                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                dists = np.load(npy_path)
                anomaly_map = dists2map(dists, image.shape)

                current_vmax = max(float(vmax[object_name]), float(np.nanmax(anomaly_map)), 1e-12)
                normalized = np.clip(anomaly_map / current_vmax, 0.0, 1.0)
                overlay_rgba = cmap(normalized)
                overlay_rgb = overlay_rgba[..., :3]
                overlay_alpha = overlay_rgba[..., 3:4]
                image_float = image.astype(np.float32) / 255.0
                blended = image_float * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
                blended = np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)

                legacy_output_path = os.path.splitext(npy_path)[0] + ".png"
                if os.path.exists(legacy_output_path):
                    os.remove(legacy_output_path)

                stem = os.path.splitext(filename)[0] + ".png"
                overlay_output_path = os.path.join(root, "Overlays_png", stem)
                patch_grid_output_path = os.path.join(root, "Patch-Gitter_png", stem)

                os.makedirs(os.path.dirname(overlay_output_path), exist_ok=True)
                Image.fromarray(blended).save(overlay_output_path)
                exported_overlays += 1

                _save_patch_distance_grid_png(dists, patch_grid_output_path)
                exported_patch_grids += 1

    print(
        f"Saved {exported_overlays} overlay PNG(s) and {exported_patch_grids} patch-grid PNG(s) "
        f"in subfolders under {anomaly_maps_dir}"
    )


def create_sample_plots(experiment_path, anomaly_maps_dir, seed, dataset, data_root):
    # infer objects and anomalies, preprocessing does not matter
    objects, object_anomalies, _, _ = get_dataset_info(
        dataset,
        preprocess="informed",
        data_path=data_root)
    # infer vmax for each object
    vmax = infer_vmax(anomaly_maps_dir, objects)

    for object_name in tqdm(objects, desc="Plot anomaly maps"):
        n = len(object_anomalies[object_name])
        fig, axs = plt.subplots(n + 1, 5, figsize=(2 * 5, 2* (n + 1)))

        for i, anomaly_type in enumerate(object_anomalies[object_name]):
            # plot five test samples with anomaly maps
            anomaly_dir = os.path.join(data_root, object_name, "test", anomaly_type)
            recursive_test_scan = any(os.path.isdir(os.path.join(anomaly_dir, entry))
                                      for entry in os.listdir(anomaly_dir))
            first_five_samples = list_image_files(anomaly_dir, recursive=recursive_test_scan)[:5]
            for j, img_nr in enumerate(first_five_samples):
                image_test, gt_mask, anomaly_map = get_test_gt_map(object_name, anomaly_type,
                                                                    img_nr, anomaly_maps_dir, dataset = dataset, data_root = data_root)
                plot_sample(image_test, anomaly_map, axs[i, j], cmap=cmap, vmax=vmax[object_name])
                axs[i, j].axis('off')
                if j == 2:
                    axs[i, j].set_title(f"anomaly type: {anomaly_type}")

        first_five_good_samples = list_image_files(os.path.join(data_root, object_name, "test", "good"))[:5]
        for j, img_nr in enumerate(first_five_good_samples):
            # plot five good test samples with anomaly maps for comparison
            image_test, anomaly_map = get_test_gt_map(object_name, "good", img_nr, 
                                                      anomaly_maps_dir, dataset = dataset, data_root = data_root, good=True)
            axs[n, j].imshow(image_test)
            axs[n, j].imshow(anomaly_map, cmap=cmap, vmin=0.0,
                             vmax=max(float(vmax[object_name]), float(np.nanmax(anomaly_map)), 0.0))
            axs[n, j].axis('off')
            if j == 2:
                axs[n, j].set_title(f"good test samples (for comparison)")

        plt.tight_layout()
        plt.savefig(f"{experiment_path}/{object_name}/anomaly_maps_examples_seed={seed}.png")
        plt.close()
