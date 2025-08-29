"""
Batch‑processing script for Pythia checkpoints
---------------------------------------------
Runs *exactly* the same activation‑recording ➜ time‑series ➜ PhiID pipeline you
already tested, but loops over every training snapshot from **step000000** to
**step143000** in 1 k increments.

The script expects your usual Hydra config folder (`../config`) and re‑uses all
paths that depend on `cfg.model.revision`.  If your paths are written with
`${model.revision}` interpolations (recommended), each checkpoint’s results
will automatically land in its own sub‑directory, e.g.:

    outputs/step050000/activations/
    outputs/step050000/phyid/

Adjust the `CHECKPOINT_RANGE` or CLI args as you like.  Heavy job – make sure
you have the storage and GPU availability.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path
from typing import List, Any, Union

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- project‑local imports ----------------------------------------------------
from src.utils import perturb_model
from src.activation_recorder import ActivationRecorder, MultiPromptActivations
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition import MultiPromptPhyID

def _heatmap_synred_vs_step(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    plot_dir: Union[str, None],
    title: str = "Synergy–redundancy rank across training",
    fname: str = "syn_minus_red_heatmap.png",
    data_fname: str = "syn_minus_red_heatmap.nc",
) -> None:
    """
    Build a (n_layers × n_steps) matrix from `curves` and plot a heatmap.

    curves: step -> (layers_x, values_y) with values in [0, 1] (your current normalisation).
            Red = high (synergy), Blue = low (redundancy).
    """
    if not curves:
        print("[heatmap] nothing to plot")
        return

    # Sort steps for consistent left→right ordering
    steps_sorted = sorted(curves.keys())
    # Assume all checkpoints use the same layer indexing (they should)
    ref_layers = curves[steps_sorted[0]][0]
    n_layers = len(ref_layers)

    # Build matrix H[layer, step_idx]
    H = np.zeros((n_layers, len(steps_sorted)), dtype=np.float32)
    for j, step in enumerate(steps_sorted):
        xs, ys = curves[step]
        # Safety: verify layers line up; if not, realign by index
        if len(xs) != n_layers or not np.all(xs == ref_layers):
            # Fallback: map by layer id → value
            val_by_layer = {int(x): float(y) for x, y in zip(xs, ys)}
            H[:, j] = [val_by_layer.get(int(l), np.nan) for l in ref_layers]
        else:
            H[:, j] = ys

    # Optional: if any NaNs crept in, fill with column means
    if np.isnan(H).any():
        col_means = np.nanmean(H, axis=0)
        inds = np.where(np.isnan(H))
        H[inds] = np.take(col_means, inds[1])

    plt.figure(figsize=(12, 7))
    # RdBu_r → blue = low, red = high
    im = plt.imshow(
        H,
        origin="lower",              # layer 0 at bottom
        aspect="auto",
        cmap="RdBu_r",
        vmin=0.0, vmax=1.0,          # your [0,1] normalisation
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Normalised (synergy − redundancy) rank", rotation=90)

    # X ticks at a manageable count
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    ax.set_xticks(range(len(steps_sorted)))
    ax.set_xticklabels([f"{s:,}" for s in steps_sorted], rotation=45, ha="right")

    # Y ticks: show every k-th layer to avoid clutter
    k = max(1, n_layers // 20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=min(20, n_layers)))
    ax.set_yticks(range(0, n_layers, k))
    ax.set_yticklabels([str(int(l)) for l in ref_layers[::k]])

    plt.xlabel("Training step")
    plt.ylabel("Source layer")
    plt.title(title)
    plt.tight_layout()

    out_dir = None
    if plot_dir:
        out_dir = os.path.join(plot_dir, "checkpoint_comparison")
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=300)
        print(f"→ saved heatmap to {fpath}")
        plt.close()
    else:
        plt.show()

    # Also persist the matrix as an xarray DataArray
    try:
        da = xr.DataArray(
            H,
            dims=("source_layer", "training_step"),
            coords={
                "source_layer": ref_layers,
                "training_step": steps_sorted,
            },
            name="syn_minus_red_norm",
            attrs={"desc": "Normalised (synergy − redundancy) rank"},
        )
        if plot_dir:
            data_path = os.path.join(out_dir, data_fname)
        else:
            data_path = data_fname
        da.to_netcdf(data_path)
        print(f"→ saved heatmap data to {data_path}")
    except Exception as e:
        print(f"[warning] could not save heatmap data: {e}")



def _overlay_plot(
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        plot_dir: Union[str, None],
        ylabel: str,
        title: str,
        fname: str,
        ylims: tuple[float, float] | None = None,
    ) -> None:
        plt.figure(figsize=(10, 6))

        # deterministic colour palette (sorted by training step)
        cmap = plt.cm.get_cmap("viridis_r", len(curves))
        for i, step in enumerate(sorted(curves)):
            xs, ys = curves[step]
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=f"step {step:,}",
                color=cmap(i),
            )

        if ylims is not None:
            plt.ylim(*ylims)
        plt.xlabel("Source layer")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()

        if plot_dir:
            out_dir = os.path.join(plot_dir, "checkpoint_comparison")
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, fname)
            plt.savefig(fpath, dpi=300)
            print(f"→ saved to {fpath}")
            plt.close()
        else:
            plt.show()


def _plot_synergy_vs_step(
    totals: dict[int, float],
    plot_dir: Union[str, None],
    ylabel: str = "Total synergy  Σ(sts)",
    title: str = "Total synergy across training",
    fname: str = "total_synergy_vs_step.png",
) -> None:
    plt.figure(figsize=(10, 6))

    steps_sorted = sorted(totals)
    ys = [totals[s] for s in steps_sorted]

    plt.plot(steps_sorted, ys, marker="o", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if plot_dir:
        out_dir = os.path.join(plot_dir, "checkpoint_comparison")
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=300)
        print(f"→ saved to {fpath}")
        plt.close()
    else:
        plt.show()


def plot_for_checkpoints(
    steps: List[int],
    base_cfg: Any,
    *,
    plot_dir: Union[str, None] = None,
) -> None:
    """
    Compare checkpoints by overlaying their per‑layer curves in a single figure.

    Parameters
    ----------
    steps      : list[int]
        Training steps to load (e.g. [1000, 2000, 4000, …]).
    base_cfg   : OmegaConf
        The *base* Hydra config.  A deep copy is made for every checkpoint.
    plot_dir   : str | None
        Where to save the comparison figures.  If None → `plt.show()`.
    """
    # --- lazily import heavy deps ------------------------------------------------

    # Containers for the curves
    synergy_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    synred_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total_synergy: dict[int, float] = {}      

    # --------------------------------------------------------------------------
    # Loop over checkpoints – load the saved Φ‑ID results, extract the curves
    # --------------------------------------------------------------------------
    for step in steps:
        try:
            cfg = OmegaConf.create(base_cfg)
            cfg.model.revision = f"step{step}"
            cfg.model.shortcode = f"P-1-{step}"
            cfg.model.it = f"base/steps/base-{step}"

            phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=cfg.paths.data_phyid_dir)
            phyid.build_data_array()

            print(f"✓ step{step} loaded")
            dims_to_reduce = [
                d for d in phyid.data_array.dims if d not in ("atom", "source_layer")
            ]
            synergy_series = (
                phyid.data_array.sel(atom="sts").mean(dim=dims_to_reduce)
            )  # dims → ('source_layer',)

            xs = synergy_series.coords["source_layer"].values
            ys = synergy_series.values
            synergy_curves[step] = (xs, ys)

            # ------------------------------------------------------------------
            # 2)  Normalised (synergy – redundancy) rank per layer
            # ------------------------------------------------------------------
            rank_da = phyid.syn_minus_red_rank  # dims → ('source_node', 'source_layer')
            layer_mean = rank_da.mean(dim="source_node")
            v_min, v_max = layer_mean.min().item(), layer_mean.max().item()
            norm = (
                xr.zeros_like(layer_mean)
                if v_min == v_max
                else (layer_mean - v_min) / (v_max - v_min)
            )

            xs_nr = norm.coords["source_layer"].values
            ys_nr = norm.values
            synred_curves[step] = (xs_nr, ys_nr)

            # --- 3) *total* synergy (new) -------------------------------------
            tot = float(phyid.data_array.sel(atom="sts").sum().item())
            total_synergy[step] = tot                         #  << NEW <<
            print(f"✓ step{step} loaded  –  Σsts = {tot:.3g}")

            print(f"✓ step{step} loaded")
        except Exception as e:
            print(f"[warning] step{step} skipped: {e}")
            continue

    # --------------------------------------------------------------------------
    # Draw the two overlay figures
    # --------------------------------------------------------------------------
    cfg = OmegaConf.create(base_cfg)
    cfg.model.shortcode = f"P‑1"
    cfg.model.it = f"base"


    if synergy_curves:
        _overlay_plot(
            synergy_curves,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
            ylabel="Mean synergy (sts)",
            title="Evolution of mean synergy per layer",
            fname="synergy_per_layer_comparison.png",
        )

    if synred_curves:
        _overlay_plot(
            synred_curves,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
            ylabel="Normalised (synergy – redundancy) rank",
            title="Evolution of synergy–redundancy rank per layer",
            fname="syn_minus_red_per_layer_comparison.png",
            ylims=(-0.05, 1.05),
        )
    if total_synergy:                                   
        _plot_synergy_vs_step(
            total_synergy,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
        )
    
    # --- NEW: heatmap of (synergy − redundancy) rank across steps -------------
    if synred_curves:
        _heatmap_synred_vs_step(
            synred_curves,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
            title="Synergy–redundancy rank per layer across training",
            fname="syn_minus_red_heatmap.png",
            data_fname="syn_minus_red_heatmap.nc",
        )



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    steps = [2**i for i in range(0, 10)] + [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
    # steps = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000] 

    plot_for_checkpoints(steps, cfg)


if __name__ == "__main__":
    main()
