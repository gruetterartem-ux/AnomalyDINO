import argparse
import math
import os
from argparse import ArgumentParser, Action 
import yaml
from tqdm import trange

import csv
import json

from src.utils import get_dataset_info 
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run, eval_classification_scores
from src.visualize import create_sample_plots, export_anomaly_map_pngs
from src.backbones import get_model


class IntListAction(Action):
    """
    Define a custom action to always return a list. 
    This allows --shots 1 to be treated as a list of one element [1]. 
    """
    def __call__(self, namespace, values):
        if not isinstance(values, list):
            values = [values]
        setattr(namespace, self.dest, values)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="Name of the backbone model. Choose from ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'vit_b_16'].")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method. Choose from ['agnostic', 'informed', 'masking_only'].")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction, help="Use GPU for FAISS kNN search. (Conda install faiss-gpu recommended, does usually not work with pip install.)")
    parser.add_argument("--shots", nargs='+', type=int, default=[1], #action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--random_ref_samples", default=False, action=argparse.BooleanOptionalAction,
                        help="Sample few-shot reference images randomly without replacement instead of using a deterministic slice.")
    parser.add_argument("--use_last_ref_samples", default=False, action=argparse.BooleanOptionalAction,
                        help="Use the last few-shot reference images from the sorted train/good list.")
    parser.add_argument("--ref_image_names", nargs="+", default=None,
                        help="Explicit train/good reference image filenames to use instead of sampled few-shot references.")
    parser.add_argument("--mask_ref_images", type=bool, default=False)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction, help="Save example plots.")
    parser.add_argument('--save_anomaly_map_pngs', default=False, action=argparse.BooleanOptionalAction,
                        help="Save a PNG anomaly map for every saved .npy anomaly map in the matching folder.")
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=False, action=argparse.BooleanOptionalAction, help="Evaluate anomaly segmentation performance.")
    parser.add_argument("--aggregation_statistics", type=str, default="meantop1p",
                        choices=["meantop1p", "max_patch_distance", "max_anomaly_map"],
                        help="Image-level aggregation for anomaly scores.")
    parser.add_argument("--inference_split", type=str, default="test",
                        choices=["test", "train"],
                        help="Dataset split to run inference on.")
    parser.add_argument("--eval_remaining_train_good", default=False, action=argparse.BooleanOptionalAction,
                        help="Evaluate on remaining train/good images plus test/good and test anomalies, excluding the selected reference images.")
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of warmup iterations, relevant when benchmarking inference time.")

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    
    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess, data_path=args.data_root)

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution)

    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")

    if args.just_seed != None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)
    
    for shot in list(args.shots):
        save_examples = args.save_examples

        results_dir = f"results_{args.dataset}/{args.model_name}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"
        
        if args.tag != None:
            results_dir += "_" + args.tag
        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)
        
        # save preprocessing setups (masking and rotation) to file
        with open(f"{results_dir}/preprocess.yaml", "w") as f:
            yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)

        # save arguments to file
        with open(f"{results_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        if args.faiss_on_cpu:
            print("Warning: Running similarity search on CPU. Consider using faiss-gpu for faster inference.")
        
        print("Results will be saved to", results_dir)
    
        for seed in seeds:
            print(f"=========== Shot = {shot}, Seed = {seed} ===========")
            
            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue
            else:
                measurements_file = results_dir + f"/measurements_seed={seed}.csv"
                with open(measurements_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time", "Label", "Evaluation_Group", "Threshold"])
                    score_cache = {}

                    for object_name in objects:
                        
                        if save_examples:
                            os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                            os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                        # CUDA warmup
                        for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                            first_image = os.listdir(f"{args.data_root}/{object_name}/train/good")[0]
                            img_tensor, grid_size = model.prepare_image(f"{args.data_root}/{object_name}/train/good/{first_image}")
                            features = model.extract_features(img_tensor)
                                         
                        anomaly_scores, time_memorybank, time_inference, ref_images, sample_metadata = run_anomaly_detection(
                                                                                model,
                                                                                object_name,
                                                                                data_root = args.data_root, 
                                                                                n_ref_samples = shot,
                                                                                object_anomalies = object_anomalies,
                                                                                plots_dir = plots_dir,
                                                                                save_examples = save_examples,
                                                                                random_ref_samples = args.random_ref_samples,
                                                                                use_last_ref_samples = args.use_last_ref_samples,
                                                                                ref_image_names = args.ref_image_names,
                                                                                knn_metric = args.knn_metric,
                                                                                knn_neighbors = args.k_neighbors,
                                                                                faiss_on_cpu = args.faiss_on_cpu,
                                                                                masking = masking_default[object_name],
                                                                                mask_ref_images = args.mask_ref_images,
                                                                                rotation = rotation_default[object_name],
                                                                                seed = seed,
                                                                                aggregation_statistics = args.aggregation_statistics,
                                                                                inference_split = args.inference_split,
                                                                                eval_remaining_train_good = args.eval_remaining_train_good,
                                                                                save_patch_dists = (args.eval_clf or args.inference_split != "test"), # keep patch maps for non-test inference runs
                                                                                save_tiffs = args.eval_segm)      # save anomaly maps as tiffs for segmentation evaluation

                        score_cache[object_name] = {
                            "scores": anomaly_scores,
                            "metadata": sample_metadata,
                        }

                        with open(f"{results_dir}/reference_images_{object_name}_seed={seed}.json", "w") as ref_file:
                            json.dump(ref_images, ref_file, indent=2)

                        object_threshold = ""
                        labels = [info.get("label") for info in sample_metadata.values() if info.get("label", "") != ""]
                        if labels and len(set(labels)) > 1:
                            scores = [anomaly_scores[sample] for sample in sample_metadata.keys()]
                            _, _, _, clf_details = eval_classification_scores(labels, scores, return_details=True)
                            if math.isfinite(clf_details["threshold"]):
                                object_threshold = f"{clf_details['threshold']:.5f}"
                        
                        # write anomaly scores and inference times to file
                        for counter, sample in enumerate(anomaly_scores.keys()):
                            anomaly_score = anomaly_scores[sample]
                            inference_time = time_inference[sample]
                            sample_info = sample_metadata.get(sample, {})
                            writer.writerow([
                                object_name,
                                sample,
                                f"{anomaly_score:.5f}",
                                f"{time_memorybank:.5f}",
                                f"{inference_time:.5f}",
                                sample_info.get("label", ""),
                                sample_info.get("group", ""),
                                object_threshold,
                            ])
                        # print(f"Mean inference time ({object_name}): {sum(time_inference.values())/len(time_inference):.5f} s/sample")                        

                # read inference times from file
                with open(measurements_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    inference_times = [float(row[4]) for row in reader]
                print(f"Finished AD for {len(objects)} objects (seed {seed}), mean inference time: {sum(inference_times)/len(inference_times):.5f} s/sample = {len(inference_times)/(sum(inference_times)+1e-10):.2f} samples/s")

                if args.eval_remaining_train_good:
                    print(f"=========== Evaluate seed = {seed} (remaining train/good + test) ===========")
                    evaluation_dict = {}
                    auroc_clf_ls = []
                    ap_clf_ls = []
                    f1_clf_ls = []

                    for object_name in objects:
                        labels = [info["label"] for _, info in score_cache[object_name]["metadata"].items()]
                        scores = [score_cache[object_name]["scores"][sample] for sample in score_cache[object_name]["metadata"].keys()]
                        auroc_clf, ap_clf, f1_clf, clf_details = eval_classification_scores(labels, scores, return_details=True)
                        print(f"{object_name}: AUROC (image-level): {auroc_clf} -- Average Precision (image-level): {ap_clf} -- F1 (image-level): {f1_clf}")

                        evaluation_dict[object_name] = {
                            "classification_AUROC": auroc_clf,
                            "classification_AP": ap_clf,
                            "classification_F1": f1_clf,
                            "classification_threshold": clf_details["threshold"],
                        }
                        auroc_clf_ls.append(auroc_clf)
                        ap_clf_ls.append(ap_clf)
                        f1_clf_ls.append(f1_clf)

                    evaluation_dict['mean_classification_au_roc'] = float(sum(auroc_clf_ls) / len(auroc_clf_ls))
                    evaluation_dict['mean_classification_ap'] = float(sum(ap_clf_ls) / len(ap_clf_ls))
                    evaluation_dict['mean_classification_f1'] = float(sum(f1_clf_ls) / len(f1_clf_ls))

                    with open(f"{results_dir}/metrics_seed={seed}.json", "w") as metric_file:
                        json.dump(evaluation_dict, metric_file, indent=4)
                    print(f"Wrote metrics to {results_dir}/metrics_seed={seed}.json")

                    save_examples = False
                    continue

                if args.inference_split != "test":
                    print(f"Finished AD without evaluation on split '{args.inference_split}', inference results saved to {results_dir}")
                    save_examples = False
                    continue

                # check wheter 'good' folder exists for testing
                for object_name in objects:
                    good_folder = f"{args.data_root}/{object_name}/test/good/"
                    if not os.path.exists(good_folder):
                        print(f"Warning: 'good' folder not found for {object_name}! No evaluation will be performed for seed {seed}.")
                        print("Finished AD without evaluation, inference results saved to", results_dir)
                        break
                else:
                    # evaluate all finished runs and create sample anomaly maps for inspection
                    print(f"=========== Evaluate seed = {seed} ===========")
                    eval_finished_run(args.dataset, 
                                    args.data_root, 
                                    anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                    output_dir = results_dir,
                                    seed = seed,
                                    pro_integration_limit = 0.3,
                                    eval_clf = args.eval_clf,
                                    eval_segm = args.eval_segm,
                                    aggregation_statistics = args.aggregation_statistics)
                    
                    create_sample_plots(results_dir, 
                                        anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                        seed = seed,
                                        dataset = args.dataset, 
                                        data_root = args.data_root)

                    if args.save_anomaly_map_pngs:
                        export_anomaly_map_pngs(
                            anomaly_maps_dir=results_dir + f"/anomaly_maps/seed={seed}",
                            data_root=args.data_root,
                        )
                
                    # deactivate creation of examples for the next seeds...
                    save_examples = False 

    print("Finished and evaluated all runs!")
