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
        try:
            vals.append(float(tok))
        except ValueError:
            # try scientific like 1e-3
            vals.append(float(tok))
    return vals


def gen_logspace(lo: float, hi: float, num: int):
    return list(np.logspace(np.log10(lo), np.log10(hi), num=num))


def parse_two_floats(s: str, default=(0.45, 1.0)):
    try:
        parts = [float(x.strip()) for x in s.split(',') if x.strip()]
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return default


# Experimental: reposition Z axis relative to X/Y using private mplot3d APIs.
# Note: This relies on _axinfo and may vary across Matplotlib versions.
def _set_z_axis_anchor(ax, anchor: str):
    try:
        if anchor == "left":
            # Favor the Z axis being drawn on the left edge
            ax.zaxis._axinfo["juggled"] = (1, 2, 0)
        elif anchor == "right":
            ax.zaxis._axinfo["juggled"] = (2, 1, 0)
    except Exception:
        # Silently ignore if backend/version doesn't support this
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
    # Search last occurrence
    last_match = None
    for pat in METRIC_PATTERNS:
        for m in pat.finditer(text):
            last_match = m
    if last_match is not None:
        if 'ACC' in last_match.re.pattern.upper() and last_match.lastindex >= 2:
            # First group ACC then NMI
            try:
                acc = float(last_match.group(1))
                nmi = float(last_match.group(2))
            except Exception:
                pass
        else:
            # First group NMI then ACC
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
    # Prefer scientific form for exact powers of ten, e.g., 1e-4, 1e-1, 1e1, 1e3
    if a > 0:
        k = int(round(np.log10(a)))
        if np.isclose(a, 10.0 ** k, rtol=1e-6, atol=1e-12):
            if k == 0:
                return "1"
            return f"1e{k}"
    # For very small or very large, use scientific without + and leading zeroes in exponent
    if a < 1e-3 or a >= 1e3:
        s = f"{v:.0e}".replace("+", "")
        s = re.sub(r"e(-?)0+", r"e\1", s)
        return s
    # Otherwise use compact decimal
    if a < 1:
        return f"{v:.3f}".rstrip('0').rstrip('.')
    return f"{v:.2f}".rstrip('0').rstrip('.')


def run_one(train_script: str, dataset: str, overrides: dict, extra_args: Optional[list]):
    # Run train.py in-process with overridden args
    orig_argv = sys.argv[:]
    catcher = StdoutCatcher()
    sys.stdout = catcher
    try:
        # Ensure train.py respects CLI params by setting an env flag.
        os.environ["MVC_FORCE_PARAMS"] = "1"
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
    parser = argparse.ArgumentParser(description="3D parameter sensitivity analysis (non-invasive)")
    parser.add_argument("--dataset", default="Cifar100", type=str)
    parser.add_argument("--train_script", default="train.py", type=str)
    parser.add_argument("--metric", default="ACC", choices=["ACC", "NMI"], help="Metric to plot")
    parser.add_argument("--fixed_param", default="param3", choices=["param1", "param2", "param3"], help="Parameter to hold fixed")
    parser.add_argument("--fixed_value", default=0.0001, type=float)
    parser.add_argument("--x_param", default="param1", choices=["param1", "param2", "param3"])
    parser.add_argument("--y_param", default="param2", choices=["param1", "param2", "param3"])
    # Value sources: either explicit list or logspace
    parser.add_argument("--x_values", default="1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3", type=str)
    parser.add_argument("--y_values", default="1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3", type=str)
    parser.add_argument("--repeats", default=1, type=int, help="Repeat runs per grid cell and average")
    parser.add_argument("--aggregate", default="max", choices=["mean", "max"], help="How to aggregate repeats per grid cell")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to train.py")
    parser.add_argument("--out_fig", default="figures/param_sensitivity_3d.pdf", type=str)
    parser.add_argument("--out_csv", default="figures/param_sensitivity_3d.csv", type=str)
    # Styling controls to match paper-like plots
    parser.add_argument("--axis_label_style", default="lambda", choices=["param", "lambda"], help="Use param names or λ1/λ2/λ3 labels")
    parser.add_argument("--show_colorbar", action="store_true", help="Show colorbar (hidden by default to match paper style)")
    parser.add_argument("--show_title", action="store_true", help="Show suptitle (hidden by default)")
    parser.add_argument("--zmin", default=0.0, type=float, help="Z axis min (e.g., ACC lower bound)")
    parser.add_argument("--zmax", default=1.0, type=float, help="Z axis max (e.g., ACC upper bound)")
    parser.add_argument("--ztick_step", default=0.2, type=float, help="Z axis tick step")
    parser.add_argument("--labelpad_x", default=18, type=int, help="Padding between X label and ticks")
    parser.add_argument("--labelpad_y", default=18, type=int, help="Padding between Y label and ticks")
    parser.add_argument("--labelpad_z", default=8, type=int, help="Padding between Z label and ticks")
    parser.add_argument("--tick_fontsize", default=9, type=int, help="Tick label font size")
    parser.add_argument("--elev", default=30, type=float, help="3D view elevation")
    parser.add_argument("--azim", default=-40, type=float, help="3D view azimuth; negative values tend to put Z on the left")
    parser.add_argument("--bar_width", default=0.8, type=float, help="Bar width in both X and Y (0-1]")
    parser.add_argument("--tickpad_x", default=0, type=int, help="Padding (pt) between X tick labels and axis")
    parser.add_argument("--tickpad_y", default=0, type=int, help="Padding (pt) between Y tick labels and axis")
    parser.add_argument("--tickpad_z", default=0, type=int, help="Padding (pt) between Z tick labels and axis")
    parser.add_argument("--z_left", action="store_true", help="Force Z ticks to appear on the left by setting azim to -80")
    parser.add_argument("--z_anchor", default="left", choices=["auto", "left", "right"], help="Experimentally anchor Z axis to left/right edge")
    parser.add_argument("--proj_type", default="ortho", choices=["ortho", "persp"], help="3D projection type: orthographic or perspective")
    # Coloring controls
    parser.add_argument("--color_mode", default="ycolormap", choices=["zmap", "group", "ycolormap"], help="zmap: by ACC; group: per-y hue with ACC brightness; ycolormap: colormap along Y blended with ACC brightness")
    parser.add_argument("--cmap", default="viridis", type=str, help="Colormap name used when color_mode=zmap")
    parser.add_argument("--group_hues", default="0.62,0.13", type=str, help="Hue range [start,end] in HSV for y-groups (blue->yellow by default)")
    parser.add_argument("--value_gamma", default=1.0, type=float, help="Gamma for ACC-to-brightness mapping (group mode)")
    parser.add_argument("--value_range", default="0.45,1.0", type=str, help="Brightness range [v_min,v_max] for group mode")
    parser.add_argument("--y_cmap", default="viridis", type=str, help="Colormap along Y when color_mode=ycolormap")
    parser.add_argument("--y_blend", default=0.4, type=float, help="Blend factor (0-1) mixing ACC brightness into Y colormap in ycolormap mode")
    parser.add_argument("--y_value_range", default="0.45,1.0", type=str, help="Brightness [v_min,v_max] in ycolormap mode")
    parser.add_argument("--y_gamma", default=1.0, type=float, help="Gamma for ACC->brightness in ycolormap mode")
    args = parser.parse_args()

    if args.fixed_param in (args.x_param, args.y_param):
        raise ValueError("fixed_param must be different from x_param and y_param")

    x_vals = parse_floats_list(args.x_values)
    y_vals = parse_floats_list(args.y_values)
    Xn, Yn = len(x_vals), len(y_vals)

    Z = np.full((Yn, Xn), np.nan, dtype=np.float32)  # row=y, col=x

    for iy, y in enumerate(y_vals):
        for ix, x in enumerate(x_vals):
            acc_list, nmi_list = [], []
            for r in range(max(1, int(args.repeats))):
                overrides = {
                    args.x_param: x,
                    args.y_param: y,
                    args.fixed_param: args.fixed_value,
                }
                acc, nmi, _ = run_one(args.train_script, args.dataset, overrides, args.extra_args)
                if acc is not None:
                    acc_list.append(acc)
                if nmi is not None:
                    nmi_list.append(nmi)
            # choose metric + aggregation
            if args.metric == "ACC":
                if acc_list:
                    val = (np.mean(acc_list) if args.aggregate == "mean" else np.max(acc_list))
                else:
                    val = np.nan
            else:
                if nmi_list:
                    val = (np.mean(nmi_list) if args.aggregate == "mean" else np.max(nmi_list))
                else:
                    val = np.nan
            Z[iy, ix] = val

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', encoding='utf-8') as f:
        header = ["y_param"] + [format_tick(v) for v in x_vals]
        f.write(",".join(header) + "\n")
        for iy, y in enumerate(y_vals):
            row = [format_tick(y)] + ["" if np.isnan(Z[iy, ix]) else f"{Z[iy, ix]:.4f}" for ix in range(Xn)]
            f.write(",".join(row) + "\n")

    # Plot 3D bar chart
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    # Apply projection type (ortho helps Z look vertical when changing elevation)
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

    # Build colors
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = []
    if args.color_mode == "zmap":
        # Colormap driven purely by ACC (Z)
        cmap = cm.get_cmap(args.cmap)
        colors = cmap(norm(dz))
    elif args.color_mode == "group":
        # Group mode: per-y hue, ACC controls brightness
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
        # ycolormap: colors vary smoothly along Y from a colormap; modulate brightness by ACC
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

    # Axis labels and ticks
    def pretty_axis_label(p: str) -> str:
        if args.axis_label_style == "lambda":
            mapping = {"param1": "λ1", "param2": "λ2", "param3": "λ3"}
            return mapping.get(p, p)
        return p

    ax.set_xlabel(pretty_axis_label(args.x_param), labelpad=args.labelpad_x)
    ax.set_ylabel(pretty_axis_label(args.y_param), labelpad=args.labelpad_y)
    ax.set_zlabel(args.metric, labelpad=args.labelpad_z)
    # Put ticks at bar centers
    ax.set_xticks(np.arange(Xn) + bw/2)
    ax.set_yticks(np.arange(Yn) + bw/2)
    ax.set_xticklabels([format_tick(v) for v in x_vals], rotation=35, ha='right', fontsize=args.tick_fontsize)
    ax.set_yticklabels([format_tick(v) for v in y_vals], rotation=0, fontsize=args.tick_fontsize)
    ax.tick_params(axis='x', pad=args.tickpad_x)
    ax.tick_params(axis='y', pad=args.tickpad_y)
    ax.tick_params(axis='z', pad=args.tickpad_z)
    # Tight axis limits so bars are fully wrapped by axes [0, (N-1)+bar_width]
    ax.set_xlim(0, (Xn - 1) + bw)
    ax.set_ylim(0, (Yn - 1) + bw)
    # Z axis range and ticks to [0,1] with 0.2 step by default
    try:
        zmin, zmax = float(args.zmin), float(args.zmax)
    except Exception:
        zmin, zmax = 0.0, 1.0
    ax.set_zlim(zmin, zmax)
    if args.ztick_step > 0:
        ticks = np.arange(zmin, zmax + 1e-9, args.ztick_step)
        ax.zaxis.set_major_locator(FixedLocator(ticks))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # A cleaner paper-like style: adjust view so Z ticks appear on the left by default
    # If requested, force Z ticks on the left by overriding azim
    azim = -80 if args.z_left else float(args.azim)
    ax.view_init(elev=float(args.elev), azim=azim)
    if args.z_anchor != "auto":
        _set_z_axis_anchor(ax, args.z_anchor)
    # Optional colorbar and title
    if args.show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(args.metric)
    if args.show_title:
        fig.suptitle(
            f"{args.dataset} — {args.metric} vs {pretty_axis_label(args.x_param)} & {pretty_axis_label(args.y_param)} "
            f"(fixed {pretty_axis_label(args.fixed_param)}={format_tick(args.fixed_value)})"
        )

    plt.tight_layout()
    plt.savefig(args.out_fig, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {args.out_fig}")
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
