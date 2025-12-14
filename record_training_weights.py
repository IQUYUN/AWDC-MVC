import os
import sys
import re
import runpy
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MultipleLocator, MaxNLocator


def vivid_colors(n: int):
    n = max(1, int(n))
    hues = np.linspace(0.0, 1.0, num=n, endpoint=False)
    s, v = 0.90, 0.95
    return [tuple(hsv_to_rgb([h, s, v])) for h in hues]


def average_weight_tensor(weight, view: int):
    import torch
    if weight is None:
        return np.full((view,), 1.0 / view, dtype=np.float32)
    wt = weight
    if not isinstance(wt, torch.Tensor):
        wt = torch.as_tensor(wt)
    if wt.shape[-1] != view:
        dims = list(wt.shape)
        if view in dims:
            idx = dims.index(view)
            wt = wt.movedim(idx, -1)
        else:
            return np.full((view,), 1.0 / view, dtype=np.float32)
    while wt.dim() > 1:
        wt = wt.mean(dim=0)
    wt = wt.detach().cpu().float()
    s = wt.sum()
    if abs(float(s)) > 1e-8:
        wt = wt / s
    return wt.numpy()


class StdoutEpochSniffer:
    """Wrap sys.stdout to detect lines like 'Epoch 12' and trigger a callback(epoch_num, line)."""
    def __init__(self, on_epoch_line):
        self._orig = sys.stdout
        self._buf = ''
        self._on_epoch_line = on_epoch_line
        self._pat = re.compile(r"Epoch\s*(\d+)")

    def write(self, s):
        self._orig.write(s)
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            m = self._pat.search(line)
            if m:
                try:
                    ep_num = int(m.group(1))
                except Exception:
                    ep_num = None
                self._on_epoch_line(ep_num, line)

    def flush(self):
        self._orig.flush()


def get_view_names(dataset: str, view: int, dims=None):
    """Return human-friendly view names with dimensions when available.
    - Caltech-5V: WM/CENTRIST/LBP/GIST/HOG with dims e.g., WM(40)
    - Caltech-* (2V/3V/4V): base names with dims when provided
    - Others: View i (dim)
    """
    dims_list = None
    try:
        if dims is not None:
            dims_list = [int(d) for d in list(dims)]
    except Exception:
        dims_list = None

    if dataset == "Caltech-5V" and view == 5:
        base = ["WM", "CENTRIST", "LBP", "GIST", "HOG"]
        if dims_list and len(dims_list) == 5:
            return [f"{base[i]}({dims_list[i]})" for i in range(5)]
        return base[:view]
    if dataset.startswith("Caltech-"):
        base = ["WM", "CENTRIST", "LBP", "GIST", "HOG"]
        names = []
        for i in range(view):
            name = base[i] if i < len(base) else f"View {i+1}"
            if dims_list and i < len(dims_list):
                name = f"{name}({dims_list[i]})"
            names.append(name)
        return names
    # default: View i (dim)
    names = []
    for i in range(view):
        if dims_list and i < len(dims_list):
            names.append(f"View {i+1}({dims_list[i]})")
        else:
            names.append(f"View {i+1}")
    return names


def plot_weights(weights, out_path, dataset, view_names,
                 ymin=0.0, ymax=0.5, line_width=1.2,
                 figsize=(7.0, 5.0), xtick_step=0, xtick_num=8,
                 ensure_epoch=None):
    weights = np.asarray(weights, dtype=np.float32)
    epochs = weights.shape[0]
    view = weights.shape[1]
    colors = vivid_colors(view)
    plt.figure(figsize=figsize)
    x = np.arange(1, epochs + 1)
    # Define a set of markers and linestyles to help distinguish views beyond color
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    # Place markers sparsely along the line to avoid clutter
    markevery = max(1, epochs // 10)
    for v in range(view):
        marker = markers[v % len(markers)]
        linestyle = linestyles[v % len(linestyles)]
        plt.plot(x, weights[:, v], label=view_names[v], color=colors[v], linewidth=line_width,
                 linestyle=linestyle, marker=marker, markersize=4, markevery=markevery)
    plt.xlabel("Epoch")
    plt.ylabel("Weight Value")
    plt.title(f"Adaptive View Weights during Original Training — {dataset}")
    plt.ylim(ymin, ymax)
    # X ticks density control
    ax = plt.gca()
    if xtick_step and xtick_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(max(1, int(xtick_step))))
    else:
        # Let Matplotlib decide nice integer ticks with an upper bound on count
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max(2, int(xtick_num)), integer=True))
    # Ensure a specific epoch tick (e.g., 50) is shown
    if ensure_epoch is not None and 1 <= int(ensure_epoch) <= epochs:
        curr = list(ax.get_xticks())
        # Keep ticks in [1, epochs] and integerize
        curr_int = [int(round(t)) for t in curr if 1 <= t <= epochs]
        if int(ensure_epoch) not in curr_int:
            curr_int.append(int(ensure_epoch))
        curr_int = sorted(set(curr_int))
        ax.set_xticks(curr_int)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="best")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Record adaptive weights from original training (non-invasive)")
    parser.add_argument("--dataset", default="Caltech-5V", type=str)
    parser.add_argument("--train_script", default="train.py", type=str, help="Path to the original training script")
    parser.add_argument("--fig_path", default="figures/weight_dynamics_original.pdf", type=str)
    parser.add_argument("--csv_path", default="figures/weight_dynamics_original.csv", type=str)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Extra CLI args forwarded to train.py, e.g. --batch_size 256")
    parser.add_argument("--skip_epochs", default=50, type=int, help="Skip first N epochs (e.g., pretraining)")
    parser.add_argument("--ymin", default=0.0, type=float, help="Y-axis min for the plot")
    parser.add_argument("--ymax", default=0.6, type=float, help="Y-axis max for the plot")
    parser.add_argument("--line_width", default=0.5, type=float, help="Line width for the plot")
    parser.add_argument("--figsize", nargs=2, type=float, default=[7.0, 5.0], help="Figure size (width height in inches)")
    parser.add_argument("--xtick_step", default=20, type=int, help="Fix x tick step (epochs). 0 = auto")
    parser.add_argument("--xtick_num", default=8, type=int, help="Approximate number of x ticks when auto")
    parser.add_argument("--ensure_epoch", default=0, type=int, help="Ensure a tick at this epoch if within range; 0=disabled")
    args = parser.parse_args()

    # Import modules we need to patch
    import importlib
    net_mod = importlib.import_module('network')
    data_mod = importlib.import_module('dataloader')

    # Build a temporary model to get view count
    dataset_tuple = data_mod.load_data(args.dataset)
    dataset, dims, view, data_size, class_num = dataset_tuple
    view_names = get_view_names(args.dataset, view, dims)

    # Prepare collectors
    weights_buffer = []  # current epoch buffer (list of np vectors)
    weights_per_epoch = []

    # Patch Network.forward to tap weights
    orig_forward = net_mod.Network.forward

    def tapped_forward(self, xs, zs_gradient=True):
        xrs, zs, rs, H, weight = orig_forward(self, xs, zs_gradient=zs_gradient)
        try:
            w_vec = average_weight_tensor(weight, view)
            weights_buffer.append(w_vec)
        except Exception:
            pass
        return xrs, zs, rs, H, weight

    net_mod.Network.forward = tapped_forward

    # Patch stdout to detect epoch boundaries
    seen_epochs = 0

    def on_epoch_line(ep_num, _line):
        nonlocal seen_epochs
        # finalize current buffer for the epoch that just finished
        if weights_buffer:
            seen_epochs += 1
            if seen_epochs > args.skip_epochs:
                ep = np.stack(weights_buffer, axis=0).mean(axis=0)
                weights_per_epoch.append(ep)
            # clear buffer regardless
            weights_buffer.clear()

    stdout_wrapper = StdoutEpochSniffer(on_epoch_line)
    sys.stdout = stdout_wrapper

    # Prepare argv for running train.py
    orig_argv = sys.argv[:]
    try:
        sys.argv = [args.train_script, "--dataset", args.dataset]
        if args.extra_args:
            sys.argv += args.extra_args
        # Run the training script in-process as __main__
        runpy.run_path(args.train_script, run_name='__main__')
    finally:
        # Restore stdout and argv
        sys.stdout = stdout_wrapper._orig
        sys.argv = orig_argv
        # Restore original forward
        net_mod.Network.forward = orig_forward

    # Flush last partial epoch if any
    if weights_buffer:
        # treat as last epoch tail
        seen_epochs += 1
        if seen_epochs > args.skip_epochs:
            ep = np.stack(weights_buffer, axis=0).mean(axis=0)
            weights_per_epoch.append(ep)
        weights_buffer.clear()

    if not weights_per_epoch:
        print("[warn] No epoch boundaries detected; no weights recorded.")
        return

    weights_np = np.stack(weights_per_epoch, axis=0)

    # Save CSV
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    header = ",".join(["epoch"] + view_names)
    rows = [",".join([str(i+1)] + [f"{weights_np[i, j]:.6f}" for j in range(weights_np.shape[1])]) for i in range(weights_np.shape[0])]
    with open(args.csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")

    # Plot
    plot_weights(weights_np, args.fig_path, args.dataset, view_names,
                 ymin=args.ymin, ymax=args.ymax,
                 line_width=args.line_width,
                 figsize=(args.figsize[0], args.figsize[1]),
                 xtick_step=args.xtick_step, xtick_num=args.xtick_num,
                 ensure_epoch=args.ensure_epoch)
    print(f"[saved] {args.fig_path}")
    print(f"[saved] {args.csv_path}")


if __name__ == "__main__":
    main()
