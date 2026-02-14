"""
Run PVT_CASCADE on test dataset(s), print test_dice, and save grid plots:
8 columns = original | ground truth | level1 | level2 | level3 | level4 | final prediction | depth.
Rows sorted by dice (ascending, worst first), max 20 rows per figure.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from CASCADE.networks import PVT_CASCADE
from utils.dataloader import test_dataset
from utils.extend_CASCADE import extend_CASCADE_classifier


# ImageNet normalization used by test_dataset
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Default test datasets (only these when recursive_test)
DEFAULT_TEST_DATASETS = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")


def get_proper_device(device: str):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device(device)


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) normalized -> (H, W, 3) uint8."""
    x = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return x


def pred_to_np(pred: torch.Tensor, size: tuple) -> np.ndarray:
    """Upsample prediction to size and return (H, W) in [0, 1]."""
    p = F.upsample(pred, size=size, mode="bilinear", align_corners=False)
    p = p.sigmoid().data.cpu().numpy().squeeze()
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    return p


def load_depth_image(depth_dataset_root: str | None, dataset_name: str, name: str) -> np.ndarray | None:
    """Load depth image from depth_dataset_root/{dataset_name}/depth/ or .../depths/. Returns (H,W) or (H,W,3) or None."""
    if not depth_dataset_root or not os.path.isdir(depth_dataset_root):
        return None
    base = os.path.join(depth_dataset_root, dataset_name)
    for folder in ("depth", "depths"):
        path = os.path.join(base, folder, name)
        if os.path.isfile(path):
            try:
                img = np.asarray(Image.open(path))
                return img
            except Exception:
                return None
    return None


def run_test_on_single_dataset(
    model: torch.nn.Module,
    data_path: str,
    dataset_name: str,
    img_size: int,
    device: torch.device,
    depth_dataset_root: str | None = None,
):
    """Run test on one dataset. Returns mean dice and list of (name, dice, orig, gt, r1, r2, r3, r4, res, depth)."""
    image_root = os.path.join(data_path, "images")
    gt_root = os.path.join(data_path, "masks")
    if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
        return float("nan"), []
    loader = test_dataset(image_root, gt_root, img_size)
    model.eval()
    smooth = 1
    results = []
    with torch.no_grad():
        for _ in range(loader.size):
            image, gt, name = loader.load_data()
            gt_np = np.asarray(gt, np.float32)
            gt_np = gt_np / (gt_np.max() + 1e-8)
            image = image.to(device)
            res1, res2, res3, res4 = model(image)
            res = F.upsample(
                res1 + res2 + res3 + res4,
                size=gt_np.shape,
                mode="bilinear",
                align_corners=False,
            )
            res_np = pred_to_np(res, gt_np.shape)
            input_flat = np.reshape(res_np, (-1))
            target_flat = np.reshape(gt_np, (-1))
            intersection = input_flat * target_flat
            dice = (2 * intersection.sum() + smooth) / (res_np.sum() + gt_np.sum() + smooth)
            dice = float(dice)
            orig_np = denormalize_tensor(image)
            gt_3ch = np.stack([gt_np] * 3, axis=-1)
            r1 = pred_to_np(res1, gt_np.shape)
            r2 = pred_to_np(res2, gt_np.shape)
            r3 = pred_to_np(res3, gt_np.shape)
            r4 = pred_to_np(res4, gt_np.shape)
            depth_np = load_depth_image(depth_dataset_root, dataset_name, name)
            results.append((name, dice, orig_np, gt_3ch, r1, r2, r3, r4, res_np, depth_np))
    return (
        sum(r[1] for r in results) / len(results) if results else float("nan"),
        results,
    )


def run_test(
    model: torch.nn.Module,
    test_dataset_path: str,
    recursive_test: bool,
    img_size: int,
    device: torch.device,
    depth_dataset_root: str | None = None,
    dataset_names: tuple[str, ...] | None = None,
):
    """Run test on path; if recursive_test, on each subdir in dataset_names (default: CVC-300, CVC-ClinicDB, etc.). Returns metrics dict and flat list of (dataset_name, name, dice, ..., depth)."""
    path = test_dataset_path
    all_metrics = {}
    all_results = []
    if recursive_test and os.path.isdir(path):
        names_to_run = dataset_names if dataset_names is not None else DEFAULT_TEST_DATASETS
        subdirs = sorted(
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d in names_to_run
        )
        for name in subdirs:
            data_path = os.path.join(path, name)
            mean_dice, results = run_test_on_single_dataset(
                model, data_path, name, img_size, device, depth_dataset_root
            )
            all_metrics[f"test_dice_{name}"] = mean_dice
            for (img_name, dice, orig, gt, r1, r2, r3, r4, res, depth) in results:
                all_results.append((name, img_name, dice, orig, gt, r1, r2, r3, r4, res, depth))
        if all_metrics:
            all_metrics["test_dice_mean"] = float(np.nanmean(list(all_metrics.values())))
    else:
        mean_dice, results = run_test_on_single_dataset(
            model, path, "test", img_size, device, depth_dataset_root
        )
        all_metrics["test_dice"] = mean_dice
        for (img_name, dice, orig, gt, r1, r2, r3, r4, res, depth) in results:
            all_results.append(("test", img_name, dice, orig, gt, r1, r2, r3, r4, res, depth))
    return all_metrics, all_results


def save_grid_plots(
    results_sorted: list,
    output_dir: str,
    cols: int = 8,
    max_rows: int = 20,
):
    """Save grid figures: 8 columns (original, gt, L1, L2, L3, L4, final, depth), max 20 rows per figure."""
    os.makedirs(output_dir, exist_ok=True)
    col_titles = ["Original", "Ground truth", "Level 1", "Level 2", "Level 3", "Level 4", "Final", "Depth"]
    batch_size = max_rows
    plot_index = 0
    for start in range(0, len(results_sorted), batch_size):
        batch = results_sorted[start : start + batch_size]
        n_rows = len(batch)
        fig, axes = plt.subplots(n_rows, cols, figsize=(3 * cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        for row, item in enumerate(batch):
            _ds, name, dice, orig, gt, r1, r2, r3, r4, res, depth = item
            imgs = [orig, gt, r1, r2, r3, r4, res, depth]
            for col, (img, title) in enumerate(zip(imgs, col_titles)):
                ax = axes[row, col]
                if img is None:
                    ax.set_facecolor("k")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="w", fontsize=10)
                elif col == 0:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                if row == 0:
                    ax.set_title(title)
                elif col == 0:
                    ax.set_title(f"{name} (dice={dice:.3f})")
                ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"plot_grid_{plot_index}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")
        plot_index += 1
    return plot_index


def main():
    parser = argparse.ArgumentParser(description="Test PVT_CASCADE and save prediction grids.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_pth/PVT_CASCADE/pvt_cascade_23.pth",
        help="Path to saved model state dict",
    )
    parser.add_argument(
        "--pvt_backbone_path",
        type=str,
        default="./pretrained_pth/pvt/pvt_v2_b2.pth",
        help="Path to PVT backbone (for building model)",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="./dataset/TestDataset/",
        help="Test dataset root (or single dataset path if not recursive)",
    )
    parser.add_argument(
        "--recursive_test",
        action="store_true",
        default=True,
        help="Run on each subdir of test_dataset_path",
    )
    parser.add_argument(
        "--no_recursive_test",
        action="store_false",
        dest="recursive_test",
        help="Single dataset at test_dataset_path",
    )
    parser.add_argument("--img_size", type=int, default=352, help="Test image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output",
        help="Directory for grid plot PNGs",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=20,
        help="Max rows per grid figure",
    )
    parser.add_argument(
        "--depth_dataset_root",
        type=str,
        default=r"D:\HCMUS_ComputerScience\selfStudy\DeepLearningStuff\Paper\PolypSegmentation\code\Polyp-PVT\dataset\TestDataset",
        help="Root path where depth images live: {depth_dataset_root}/{dataset_name}/depth or .../depths",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="When recursive_test: only run on these dataset names (default: CVC-300 CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB Kvasir)",
    )
    args = parser.parse_args()

    device = get_proper_device(args.device)
    pre_defined_model = PVT_CASCADE(pvt_backbone_path=args.pvt_backbone_path)
    model = extend_CASCADE_classifier(model=pre_defined_model)
    if os.path.isfile(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found; using untrained weights.")
    model.to(device)

    dataset_names = tuple(args.datasets) if args.datasets else None
    metrics, results = run_test(
        model,
        args.test_dataset_path,
        args.recursive_test,
        args.img_size,
        device,
        depth_dataset_root=args.depth_dataset_root,
        dataset_names=dataset_names,
    )

    print("test_dice:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    if not results:
        print("No samples to plot.")
        return

    results_sorted = sorted(results, key=lambda x: x[2])
    n_plots = save_grid_plots(
        results_sorted,
        args.output_dir,
        cols=8,
        max_rows=args.max_rows,
    )
    print(f"Saved {n_plots} grid plot(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
