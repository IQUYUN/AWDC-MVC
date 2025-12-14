import os
import argparse
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb

from dataloader import load_data
from network import Network


def ensure_trained_checkpoint(dataset: str, checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        return
    print(f"[info] checkpoint not found: {checkpoint_path}. Training via train.py ...")
    cmd = ["python", "train.py", "--dataset", dataset]
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Training finished but checkpoint not found: {checkpoint_path}")


def extract_features_all_views(model: Network, device: torch.device, dataset, view: int, data_size: int,
                               eval_batch_size: int = 1024):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=min(eval_batch_size, data_size), shuffle=False)
    model.eval()

    x_lists = [[] for _ in range(view)]
    z_lists = [[] for _ in range(view)]
    r_lists = [[] for _ in range(view)]
    h_list, y_list = [], []

    with torch.no_grad():
        for xs, y, _ in loader:
            xs = [x.to(device) for x in xs]
            _, zs, rs, H, _ = model(xs)
            for v in range(view):
                x_v = xs[v].detach().cpu().numpy().reshape(xs[v].shape[0], -1)
                z_v = zs[v].detach().cpu().numpy()
                r_v = rs[v].detach().cpu().numpy()
                x_lists[v].append(x_v)
                z_lists[v].append(z_v)
                r_lists[v].append(r_v)
            h_list.append(H.detach().cpu().numpy())
            y_list.append(y.numpy().reshape(-1))

    xs_all = [np.concatenate(x_lists[v], axis=0) for v in range(view)]
    zs_all = [np.concatenate(z_lists[v], axis=0) for v in range(view)]
    rs_all = [np.concatenate(r_lists[v], axis=0) for v in range(view)]
    h_all = np.concatenate(h_list, axis=0)
    labels = np.concatenate(y_list, axis=0)
    return xs_all, zs_all, rs_all, h_all, labels


def run_tsne(X: np.ndarray, random_state: int = 42, perplexity: float = 30.0):
    n_samples = X.shape[0]
    perp = min(perplexity, max(5.0, (n_samples - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, init="pca", learning_rate="auto",
                metric="euclidean", random_state=random_state, verbose=0)
    return tsne.fit_transform(X)


def vivid_colors(n: int):
    """Generate n bright, high-saturation distinct colors using the HSV color wheel."""
    n = max(1, int(n))
    hues = np.linspace(0.0, 1.0, num=n, endpoint=False)
    s, v = 0.90, 0.95
    return [tuple(hsv_to_rgb([h, s, v])) for h in hues]


def main():
    parser = argparse.ArgumentParser(description="Post-training visualization: t-SNE of Raw/Z/R/H")
    parser.add_argument("--dataset", default="Prokaryotic", type=str)
    parser.add_argument("--view_index", default=0, type=int, help="Single view index; -1 for all views grid")
    parser.add_argument("--perplexity", default=60.0, type=float)
    parser.add_argument("--checkpoint_dir", default="models", type=str)
    parser.add_argument("--fig_path", default="figures/tsne_all_views.pdf", type=str)
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "eps", "png"], help="pdf/svg/eps=vector, png=raster")
    parser.add_argument("--analysis_path", default="figures/tsne_analysis.txt", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--train_if_missing", action="store_true")
    args = parser.parse_args()

    # Load data and model
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    device = torch.device(args.device)
    model = Network(view, dims, feature_dim=256, high_feature_dim=32, class_num=class_num, device=device).to(device)

    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.dataset}.pth")
    if args.train_if_missing:
        ensure_trained_checkpoint(args.dataset, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)

    # Extract features
    xs_all, zs_all, rs_all, h_all, labels = extract_features_all_views(model, device, dataset, view, data_size)

    # Prepare output
    root, _ext = os.path.splitext(args.fig_path)
    out_path = f"{root}.{args.format}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_kwargs = {"format": args.format, "bbox_inches": "tight", "transparent": True}
    if args.format.lower() == "png":
        save_kwargs["dpi"] = 300

    if args.view_index >= 0:
        if args.view_index >= len(zs_all):
            raise IndexError(f"view_index {args.view_index} out of range (0..{len(zs_all)-1})")
        x_sel = xs_all[args.view_index]
        z_sel = zs_all[args.view_index]
        r_sel = rs_all[args.view_index]
        emb_x = run_tsne(x_sel, perplexity=args.perplexity)
        emb_z = run_tsne(z_sel, perplexity=args.perplexity)
        emb_r = run_tsne(r_sel, perplexity=args.perplexity)
        emb_h = run_tsne(h_all, perplexity=args.perplexity)

        fig, axs = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
        unique = np.unique(labels)
        color_list = vivid_colors(len(unique))
        # Raw
        for idx, c in enumerate(unique):
            mask = labels == c
            axs[0].scatter(emb_x[mask, 0], emb_x[mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
        axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].set_xlabel(f"{args.dataset} Raw (View {args.view_index+1})")
        # Z
        for idx, c in enumerate(unique):
            mask = labels == c
            axs[1].scatter(emb_z[mask, 0], emb_z[mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
        axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_xlabel("Z (Encoder)")
        # R
        for idx, c in enumerate(unique):
            mask = labels == c
            axs[2].scatter(emb_r[mask, 0], emb_r[mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
        axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_xlabel("R (View-MLP)")
        # H
        for idx, c in enumerate(unique):
            mask = labels == c
            axs[3].scatter(emb_h[mask, 0], emb_h[mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
        axs[3].set_xticks([]); axs[3].set_yticks([]); axs[3].set_xlabel("H (Fused)")
        fig.suptitle(f"t-SNE on {args.dataset} (view {args.view_index+1}): Raw, Z, R, H", y=0.01)
        fig.savefig(out_path, **save_kwargs)
    else:
        from matplotlib import gridspec
        emb_z_list = [run_tsne(z, perplexity=args.perplexity) for z in zs_all]
        emb_r_list = [run_tsne(r, perplexity=args.perplexity) for r in rs_all]
        emb_h = run_tsne(h_all, perplexity=args.perplexity)

        rows = 3
        cols = max(1, len(zs_all))
        fig = plt.figure(figsize=(4 * cols, 9))
        gs = gridspec.GridSpec(rows, cols, figure=fig, height_ratios=[1, 1, 1], hspace=0.25, wspace=0.15)

        # Row 0: Z^v
        for v in range(cols):
            ax = fig.add_subplot(gs[0, v])
            unique = np.unique(labels)
            color_list = vivid_colors(len(unique))
            for idx, c in enumerate(unique):
                mask = labels == c
                ax.scatter(emb_z_list[v][mask, 0], emb_z_list[v][mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{args.dataset} Z^{v+1}")

        # Row 1: R^v
        for v in range(cols):
            ax = fig.add_subplot(gs[1, v])
            unique = np.unique(labels)
            color_list = vivid_colors(len(unique))
            for idx, c in enumerate(unique):
                mask = labels == c
                ax.scatter(emb_r_list[v][mask, 0], emb_r_list[v][mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{args.dataset} R^{v+1}")

        # Row 2: H spanning all columns
        ax_h = fig.add_subplot(gs[2, :])
        unique = np.unique(labels)
        color_list = vivid_colors(len(unique))
        for idx, c in enumerate(unique):
            mask = labels == c
            ax_h.scatter(emb_h[mask, 0], emb_h[mask, 1], s=6, alpha=0.8, color=color_list[idx % len(color_list)])
        ax_h.set_xticks([]); ax_h.set_yticks([])
        ax_h.set_xlabel(f"{args.dataset} H (Fused)")
        fig.suptitle(f"t-SNE on {args.dataset}: Z^v (all views), R^v (all views), H (fused)", y=0.01)
        fig.savefig(out_path, **save_kwargs)

    print(f"[saved] {out_path}")

    # Save analysis text
    os.makedirs(os.path.dirname(args.analysis_path), exist_ok=True)
    with open(args.analysis_path, "w", encoding="utf-8") as f:
        f.write(f"t-SNE figures generated for {args.dataset}.\n")
    print(f"[saved] {args.analysis_path}")


if __name__ == "__main__":
    main()
