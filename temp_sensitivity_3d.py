import os
import sys
import runpy
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D
from typing import Optional
from matplotlib.ticker import FixedLocator, FormatStrFormatter


def parse_floats_list(s: str):
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def parse_two_floats(s: str, default=(0.45, 1.0)):
    try:
        parts = [float(x.strip()) for x in s.split(',') if x.strip()]
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return default


def _set_z_axis_anchor(ax, anchor: str):
    try:
        if anchor == "left":
            ax.zaxis._axinfo["juggled"] = (1, 2, 0)
        elif anchor == "right":
            ax.zaxis._axinfo["juggled"] = (2, 1, 0)
    except Exception:
        pass


class StdoutCatcher:
    def __init__(self):
        self._orig = sys.stdout
        self.lines = []

    def write(self, s):
        self._orig.write(s)
        self.lines.append(s)

    def flush(self):
        self._orig.flush()

    def get_text(self):
        return ''.join(self.lines)


METRIC_PATTERNS = [
    re.compile(r"ACC\s*=\s*([0-9]*\.?[0-9]+)\s+NMI\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    re.compile(r"NMI\s*=\s*([0-9]*\.?[0-9]+).*ACC\s*=\s*([0-9]*\.?[0-9]+)", re.I),
]


def extract_metrics(text: str):
    acc, nmi = None, None
    last_match = None
    for pat in METRIC_PATTERNS:
        for m in pat.finditer(text):
            last_match = m
    if last_match is not None:
        if 'ACC' in last_match.re.pattern.upper() and last_match.lastindex >= 2:
            try:
                acc = float(last_match.group(1))
                nmi = float(last_match.group(2))
            except Exception:
                pass
        else:
            try:
                nmi = float(last_match.group(1))
                acc = float(last_match.group(2))
            except Exception:
                pass
    return acc, nmi


def format_tick(v: float):
    if v == 0:
        return "0"
    a = abs(v)
    if a > 0:
        k = int(round(np.log10(a)))
        if np.isclose(a, 10.0 ** k, rtol=1e-6, atol=1e-12):
            if k == 0:
                return "1"
            return f"1e{k}"
    if a < 1e-3 or a >= 1e3:
        s = f"{v:.0e}".replace("+", "")
        s = re.sub(r"e(-?)0+", r"e\1", s)
        return s
    if a < 1:
        return f"{v:.3f}".rstrip('0').rstrip('.')
    return f"{v:.2f}".rstrip('0').rstrip('.')


def run_one(train_script: str, dataset: str, overrides: dict, extra_args: Optional[list]):
    orig_argv = sys.argv[:]
    catcher = StdoutCatcher()
    sys.stdout = catcher
    try:
        os.environ["MVC_FORCE_TEMPS"] = "1"
        argv = [train_script, "--dataset", dataset]
        for k, v in overrides.items():
            argv += [f"--{k}", str(v)]
        if extra_args:
            argv += extra_args
        sys.argv = argv
        runpy.run_path(train_script, run_name='__main__')
    finally:
        sys.stdout = catcher._orig
        sys.argv = orig_argv
    text = catcher.get_text()
    acc, nmi = extract_metrics(text)
    return acc, nmi, text


def main():
    parser = argparse.ArgumentParser(description="3D temperature sensitivity analysis (non-invasive)")
    parser.add_argument("--dataset", default="Cifar100", type=str)
    parser.add_argument("--train_script", default="train.py", type=str)
    parser.add_argument("--metric", default="ACC", choices=["ACC", "NMI"], help="Metric to plot")
    parser.add_argument("--x_param", default="temperature1", choices=["temperature1", "temperature2"], help="X-axis temperature")
    parser.add_argument("--y_param", default="temperature2", choices=["temperature1", "temperature2"], help="Y-axis temperature")
    parser.add_argument("--x_values", default="0.5,0.7", type=str)
    parser.add_argument("--y_values", default="0.5,0.7,0.9", type=str)
    parser.add_argument("--repeats", default=1, type=int, help="Repeat runs per grid cell and average")
    parser.add_argument("--aggregate", default="max", choices=["mean", "max"], help="How to aggregate repeats per grid cell")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to train.py")
    parser.add_argument("--out_fig", default="figures/temp_sensitivity_3d.pdf", type=str)
    parser.add_argument("--out_csv", default="figures/temp_sensitivity_3d.csv", type=str)
    # Styling to match paper figure aesthetics
    parser.add_argument("--axis_label_style", default="lambda", choices=["param", "lambda"], help="Use τ_F/τ_L style or raw names")
    parser.add_argument("--show_colorbar", action="store_true")
    parser.add_argument("--show_title", action="store_true")
    parser.add_argument("--zmin", default=0.0, type=float)
    parser.add_argument("--zmax", default=1.0, type=float)
    parser.add_argument("--ztick_step", default=0.2, type=float)
    parser.add_argument("--labelpad_x", default=18, type=int)
    parser.add_argument("--labelpad_y", default=18, type=int)
    parser.add_argument("--labelpad_z", default=8, type=int)
    parser.add_argument("--tick_fontsize", default=9, type=int)
    parser.add_argument("--elev", default=30, type=float)
    parser.add_argument("--azim", default=-40, type=float)
    parser.add_argument("--bar_width", default=0.9, type=float, help="Set <1 to create spacing between bars; first bar starts at X=0")
    parser.add_argument("--tickpad_x", default=0, type=int)
    parser.add_argument("--tickpad_y", default=0, type=int)
    parser.add_argument("--tickpad_z", default=0, type=int)
    parser.add_argument("--z_left", action="store_true")
    parser.add_argument("--z_anchor", default="left", choices=["auto", "left", "right"])
    parser.add_argument("--proj_type", default="ortho", choices=["ortho", "persp"])
    # Coloring controls similar to y-based comfortable palette
    parser.add_argument("--color_mode", default="ycolormap", choices=["zmap", "group", "ycolormap"])
    parser.add_argument("--cmap", default="viridis", type=str)
    parser.add_argument("--group_hues", default="0.62,0.13", type=str)
    parser.add_argument("--value_gamma", default=1.0, type=float)
    parser.add_argument("--value_range", default="0.45,1.0", type=str)
    parser.add_argument("--y_cmap", default="viridis", type=str)
    parser.add_argument("--y_blend", default=0.4, type=float)
    parser.add_argument("--y_value_range", default="0.45,1.0", type=str)
    parser.add_argument("--y_gamma", default=1.0, type=float)
    args = parser.parse_args()

    x_vals = parse_floats_list(args.x_values)
    y_vals = parse_floats_list(args.y_values)
    Xn, Yn = len(x_vals), len(y_vals)

    Z = np.full((Yn, Xn), np.nan, dtype=np.float32)

    for iy, y in enumerate(y_vals):
        for ix, x in enumerate(x_vals):
            acc_list, nmi_list = [], []
            for _ in range(max(1, int(args.repeats))):
                overrides = {
                    args.x_param: x,
                    args.y_param: y,
                }
                acc, nmi, _text = run_one(args.train_script, args.dataset, overrides, args.extra_args)
                if acc is not None:
                    acc_list.append(acc)
                if nmi is not None:
                    nmi_list.append(nmi)
            if args.metric == "ACC":
                val = (np.mean(acc_list) if (acc_list and args.aggregate == "mean") else (np.max(acc_list) if acc_list else np.nan))
            else:
                val = (np.mean(nmi_list) if (nmi_list and args.aggregate == "mean") else (np.max(nmi_list) if nmi_list else np.nan))
            Z[iy, ix] = val

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', encoding='utf-8') as f:
        header = ["y_temp"] + [format_tick(v) for v in x_vals]
        f.write(",".join(header) + "\n")
        for iy, y in enumerate(y_vals):
            row = [format_tick(y)] + ["" if np.isnan(Z[iy, ix]) else f"{Z[iy, ix]:.4f}" for ix in range(Xn)]
            f.write(",".join(row) + "\n")

    # Plot 3D bar chart
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.set_proj_type(args.proj_type)
    except Exception:
        pass

    xpos, ypos = np.meshgrid(np.arange(Xn), np.arange(Yn))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    bw = max(0.05, min(1.0, float(args.bar_width)))
    dx = bw * np.ones_like(xpos)
    dy = bw * np.ones_like(ypos)
    dz = Z.flatten()

    # Colors
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = []
    if args.color_mode == "zmap":
        cmap = cm.get_cmap(args.cmap)
        colors = cmap(norm(dz))
    elif args.color_mode == "group":
        def hsv_to_rgb(h, s, v):
            return tuple(float(x) for x in matplotlib.colors.hsv_to_rgb((h, s, v)))
        h0, h1 = parse_two_floats(args.group_hues, default=(0.62, 0.13))
        v_min, v_max = parse_two_floats(args.value_range, default=(0.45, 1.0))
        hues = np.linspace(h0, h1, Yn)
        s_const = 0.85
        colors = []
        for iy in range(Yn):
            for ix in range(Xn):
                val = Z[iy, ix]
                t = norm(val) if np.isfinite(val) else 0.0
                if args.value_gamma != 1.0:
                    t = float(np.clip(t, 0.0, 1.0)) ** float(args.value_gamma)
                v = v_min + (v_max - v_min) * t
                colors.append(hsv_to_rgb(hues[iy], s_const, v))
    else:
        y_cmap = cm.get_cmap(args.y_cmap)
        v_min, v_max = parse_two_floats(args.y_value_range, default=(0.45, 1.0))
        denom = max(1, Yn - 1)
        for iy in range(Yn):
            base_rgb = y_cmap(iy / denom)
            base_h, base_s, base_v = matplotlib.colors.rgb_to_hsv(base_rgb[:3])
            for ix in range(Xn):
                val = Z[iy, ix]
                t = norm(val) if np.isfinite(val) else 0.0
                if args.y_gamma != 1.0:
                    t = float(np.clip(t, 0.0, 1.0)) ** float(args.y_gamma)
                v_acc = v_min + (v_max - v_min) * t
                v = (1.0 - float(args.y_blend)) * base_v + float(args.y_blend) * v_acc
                rgb = matplotlib.colors.hsv_to_rgb((base_h, base_s, v))
                colors.append(tuple(float(x) for x in rgb))

    ax.bar3d(xpos, ypos, zpos, dx, dy, np.nan_to_num(dz, nan=0.0), color=colors, shade=True, edgecolor='k', linewidth=0.3)

    # Labels and ticks
    def pretty_axis_label(name: str) -> str:
        if args.axis_label_style == "lambda":
            mapping = {"temperature1": "τ_F", "temperature2": "τ_L"}
            return mapping.get(name, name)
        return name

    ax.set_xlabel(pretty_axis_label(args.x_param), labelpad=args.labelpad_x)
    ax.set_ylabel(pretty_axis_label(args.y_param), labelpad=args.labelpad_y)
    ax.set_zlabel(args.metric, labelpad=args.labelpad_z)

    ax.set_xticks(np.arange(Xn) + bw/2)
    ax.set_yticks(np.arange(Yn) + bw/2)
    ax.set_xticklabels([format_tick(v) for v in x_vals], rotation=35, ha='right', fontsize=args.tick_fontsize)
    ax.set_yticklabels([format_tick(v) for v in y_vals], rotation=0, fontsize=args.tick_fontsize)
    ax.tick_params(axis='x', pad=args.tickpad_x)
    ax.tick_params(axis='y', pad=args.tickpad_y)
    ax.tick_params(axis='z', pad=args.tickpad_z)

    ax.set_xlim(0, (Xn - 1) + bw)
    ax.set_ylim(0, (Yn - 1) + bw)

    try:
        zmin, zmax = float(args.zmin), float(args.zmax)
    except Exception:
        zmin, zmax = 0.0, 1.0
    ax.set_zlim(zmin, zmax)
    if args.ztick_step > 0:
        ticks = np.arange(zmin, zmax + 1e-9, args.ztick_step)
        ax.zaxis.set_major_locator(FixedLocator(ticks))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    azim = -80 if args.z_left else float(args.azim)
    ax.view_init(elev=float(args.elev), azim=azim)
    if args.z_anchor != "auto":
        _set_z_axis_anchor(ax, args.z_anchor)

    if args.show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(args.metric)
    if args.show_title:
        fig.suptitle(
            f"{args.dataset} — {args.metric} vs {pretty_axis_label(args.x_param)} & {pretty_axis_label(args.y_param)}"
        )

    plt.tight_layout()
    plt.savefig(args.out_fig, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {args.out_fig}")
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
