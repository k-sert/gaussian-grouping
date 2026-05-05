"""
Boundary F-score evaluation.

Measures how well predicted segmentation boundaries align with GT boundaries.
Run this on two models (with / without lambda_boundary) to compare the effect
of the Stage 3 Boundary-Aware Segmentation Regularizer.

Usage:
    python script/eval_boundary.py -m <model_path> [--iteration N] \
        [--skip_train] [--skip_test] [--tolerance 2]

Output:
    Prints mean Precision / Recall / F1 and saves a JSON next to the model.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def _sobel_edges(seg_np):
    """Return a binary edge map from an integer segmentation array [H, W]."""
    seg = torch.from_numpy(seg_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    ex = F.conv2d(seg, kx, padding=1)
    ey = F.conv2d(seg, ky, padding=1)
    mag = torch.sqrt(ex ** 2 + ey ** 2).squeeze().numpy()
    return (mag > 0).astype(np.uint8)


def boundary_f_score(pred_seg, gt_seg, tolerance=2):
    """
    Boundary F-score between two integer segmentation maps.

    A predicted boundary pixel is a true positive if there is a GT boundary
    pixel within `tolerance` pixels, and vice-versa for recall.

    Returns (precision, recall, f1).
    """
    pred_b = _sobel_edges(pred_seg)
    gt_b   = _sobel_edges(gt_seg)

    n_pred = int(pred_b.sum())
    n_gt   = int(gt_b.sum())

    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0

    struct     = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_b,   structure=struct).astype(np.uint8)
    pd_dilated = binary_dilation(pred_b, structure=struct).astype(np.uint8)

    precision = float((pred_b & gt_dilated).sum()) / n_pred
    recall    = float((gt_b & pd_dilated).sum())   / n_gt

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def evaluate_split(model_path, split_name, iteration, views,
                   gaussians, pipeline, background, classifier, tolerance):
    records = []

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc=f"  {split_name}")):
            render_obj = render(view, gaussians, pipeline, background)["render_object"]
            logits     = classifier(render_obj)
            pred_seg   = torch.argmax(logits, dim=0).cpu().numpy().astype(np.int32)
            gt_seg     = view.objects.cpu().numpy().astype(np.int32)

            p, r, f1 = boundary_f_score(pred_seg, gt_seg, tolerance=tolerance)
            records.append({"view": idx, "precision": p, "recall": r, "f1": f1})

    mean_p  = float(np.mean([x["precision"] for x in records]))
    mean_r  = float(np.mean([x["recall"]    for x in records]))
    mean_f1 = float(np.mean([x["f1"]        for x in records]))

    print(f"\n  [{split_name}] BF-score (tol={tolerance}px):  "
          f"P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}")

    result = {
        "split":           split_name,
        "iteration":       iteration,
        "tolerance_px":    tolerance,
        "mean_precision":  mean_p,
        "mean_recall":     mean_r,
        "mean_f1":         mean_f1,
        "per_view":        records,
    }

    out_path = os.path.join(model_path,
                            f"boundary_eval_{split_name}_{iteration}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dataset, iteration, pipeline, skip_train, skip_test, tolerance):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    num_classes = dataset.num_classes
    classifier  = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    ckpt_path = os.path.join(
        dataset.model_path, "point_cloud",
        f"iteration_{scene.loaded_iter}", "classifier.pth"
    )
    classifier.load_state_dict(torch.load(ckpt_path))

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"\nModel : {dataset.model_path}")
    print(f"Iter  : {scene.loaded_iter}")

    if not skip_train:
        evaluate_split(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, classifier, tolerance)

    if not skip_test and len(scene.getTestCameras()) > 0:
        evaluate_split(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, classifier, tolerance)


if __name__ == "__main__":
    parser = ArgumentParser(description="Boundary F-score evaluation")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",   default=-1,  type=int)
    parser.add_argument("--skip_train",  action="store_true")
    parser.add_argument("--skip_test",   action="store_true")
    parser.add_argument("--tolerance",   default=2,   type=int,
                        help="Pixel distance for boundary matching (default: 2)")
    parser.add_argument("--quiet",       action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    run(model.extract(args), args.iteration, pipeline.extract(args),
        args.skip_train, args.skip_test, args.tolerance)
