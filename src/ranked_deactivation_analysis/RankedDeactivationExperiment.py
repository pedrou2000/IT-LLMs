from __future__ import annotations
import os

"""ranked_deactivation_experiment.py

Thin wrapper around `RankedDeactivationAnalysis` that makes it easy to run and
compare multiple deactivation schedules (e.g. original vs. shuffled ranking)
and to visualise both overall and per‑category performance divergence.

This module is intentionally lightweight—the heavy lifting lives inside
`RankedDeactivationAnalysis`.  The wrapper just instantiates the analysis
object as needed, keeps track of the results, and offers a few convenience
plotting helpers.  Adding new experimental runs is as simple as calling
`experiment.run(...)` with a new `name`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from src.ranked_deactivation_analysis import (
    RankedDeactivationAnalysis,  # noqa: F401 – imported for side effects / typing
    RankedDeactivationResults,
)

__all__ = ["RankedDeactivationExperiment"]


@dataclass
class RankedDeactivationExperiment:
    """Run one or more node‑deactivation experiments and plot the results."""

    # Arguments required to build a `RankedDeactivationAnalysis`.
    analysis_kwargs: Dict[str, Any]

    # Container for the results of each run: ``run_name -> results``.
    runs: Dict[str, RankedDeactivationResults] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def run(
        self,
        *,
        name: str,
        deactivate_k_nodes_per_iteration: int,
        max_deactivated_nodes: Optional[int] = None,
        micro_batch_size: int = 32,
        randomise_order: bool = False,
        save_file_path: Optional[str] = None,
        reverse_kl: bool = False,
    ) -> RankedDeactivationResults:
        """Run a single deactivation experiment.

        Parameters
        ----------
        name
            A unique key that will identify this run (e.g. "original",
            "shuffled").  Used as the label in plots.
        randomise_order
            If *True*, shuffle the node‑ranking before running so the nodes are
            deactivated in a random order.  The shuffling is *in‑place* on a
            copy of the ranking inside the analysis instance, so subsequent
            runs are unaffected unless `randomise_order` is re‑used.
        All other arguments are forwarded verbatim to ``RankedDeactivationAnalysis.run``.
        """

        # Build analysis instance.
        analysis = RankedDeactivationAnalysis(**self.analysis_kwargs)

        # Optionally shuffle the node ranking.
        if randomise_order:
            analysis.randomize_node_ranking()

        # Execute.
        results = analysis.run(
            deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
            max_deactivated_nodes=max_deactivated_nodes,
            micro_batch_size=micro_batch_size,
            save_file_path=save_file_path,
            reverse_kl=reverse_kl,
        )

        # Stash for later comparison.
        if name in self.runs:
            raise ValueError(f"Run name '{name}' already exists – choose a new one.")
        self.runs[name] = results
        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def run_default_and_random(
        self,
        *,
        deactivate_k_nodes_per_iteration: int,
        max_deactivated_nodes: Optional[int] = None,
        micro_batch_size: int = 32,
        n_randomised_runs: int = 1,
        reverse_kl: bool = False,
    ) -> Dict[str, RankedDeactivationResults]:
        """Run the *original* and a *randomised* deactivation order back‑to‑back."""

        self.run(
            name="original_order",
            deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
            max_deactivated_nodes=max_deactivated_nodes,
            micro_batch_size=micro_batch_size,
            randomise_order=False,
            reverse_kl=reverse_kl,
        )
        # Run the randomised order multiple times if requested.
        for i in range(n_randomised_runs):
            self.run(
                name=f"random_order_{i + 1}",
                deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
                max_deactivated_nodes=max_deactivated_nodes,
                micro_batch_size=micro_batch_size,
                randomise_order=True,
                reverse_kl=reverse_kl,
            )
        return self.runs

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------
    def plot_overall(self, plot_dir: Optional[str] = None) -> None:
        """Plot *overall* divergence curves for all stored runs."""
        if not self.runs:
            raise RuntimeError("No runs available – call .run() first.")

        plt.figure(figsize=(10, 6))
        for run_name, res in self.runs.items():
            x = res.deactivation_schedule
            y = [r.overall_performance_divergence for r in res.deactivation_results]
            plt.plot(x, y, marker="o", label=run_name)

        plt.xlabel("Number of Deactivated Nodes")
        plt.ylabel("Overall Performance Divergence (KL)")
        plt.title("Overall Performance Divergence Comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if plot_dir:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, f"overall_divergence_plot.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()

    def plot_per_category(self, plot_dir: Optional[str] = None) -> None:
        """Plot per‑category divergence curves for every stored run."""
        if not self.runs:
            raise RuntimeError("No runs available – call .run() first.")

        # Assume categories are identical across runs – grab from the first.
        first_res = next(iter(self.runs.values()))
        categories = first_res.deactivation_results[0].kl_xr.coords["category"].values

        n_categories = len(categories)
        n_cols = 2
        n_rows = (n_categories + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex="all", sharey="all")
        axes = axes.flatten()

        for idx, category in enumerate(categories):
            ax = axes[idx]
            for run_name, res in self.runs.items():
                x = res.deactivation_schedule
                y = [
                    r.divergence_per_category.sel(category=category).item()
                    for r in res.deactivation_results
                ]
                ax.plot(x, y, marker="o", label=run_name)

            ax.set_title(str(category))
            ax.set_xlabel("# Deactivated Nodes")
            ax.set_ylabel("KL")
            ax.grid(alpha=0.3)

            if idx == 0:
                ax.legend()

        # Hide any unused subplots.
        for unused_ax in axes[n_categories:]:
            unused_ax.set_visible(False)

        fig.suptitle("Per‑Category Performance Divergence Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if plot_dir:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, f"per_category_divergence_plot.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------
    def __getitem__(self, run_name: str) -> RankedDeactivationResults:
        return self.runs[run_name]

    def __iter__(self):
        return iter(self.runs.items())
