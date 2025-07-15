from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union, Tuple
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import os, pickle
import time
from datetime import timedelta
from tqdm.auto import tqdm
import pandas as pd
from functools import cached_property
import json
import pathlib



from src.utils import ModelInformation
from src.time_series_activations import PromptTimeSeries
from src.phyid_decomposition.PhyIDTimeSeries import PhyIDTimeSeries

@dataclass
class PromptPhyID:
    """Container for the PhiID time-series of a single prompt. """

    prompt_index: int
    model_info: ModelInformation
    generated_tokens: Sequence[str] = field(default_factory=list)
    phyid: Dict[Tuple[int, int, int, int], PhyIDTimeSeries] = field(default_factory=dict, init=False, repr=False) # (source_layer_index, source_node_index, target_layer_index, target_node_index) -> PhyIDTimeSeries
    data_array: Union[xr.DataArray, None] = field(default=None, init=False, repr=False)

    @property 
    def num_layers(self) -> int:
        """Return the number of layers in the model."""
        return self.model_info.num_layers
    
    @property
    def num_nodes_per_layer(self) -> int:
        """Return the number of nodes per layer in the model."""
        # Measure the number of nodes in the first layer, assuming all layers have the same number of nodes
        if self.num_layers == 0:
            return 0
        first_layer = next(iter(self.phyid.values())).source_layer_index
        return len([k for k in self.phyid if k[0] == first_layer])
    
    @property
    def num_nodes(self) -> int:
        """Return the total number of nodes across all layers in the model."""
        return self.num_layers * self.num_nodes_per_layer        

    def get_phyid(self, source_layer_index: int, source_node_index: int, target_layer_index: int, target_node_index: int) -> PhyIDTimeSeries:
        """Retrieve or create a PhyIDTimeSeries for the given indices."""
        key = (source_layer_index, source_node_index, target_layer_index, target_node_index)
        if key not in self.phyid:
            raise KeyError(f"PhyIDTimeSeries for source layer {source_layer_index}, source node {source_node_index}, target layer {target_layer_index}, target node {target_node_index} not found.")
        return self.phyid[key]
    
    @classmethod
    def from_time_series(cls, prompt_time_series: PromptTimeSeries, model_info: ModelInformation, prompt_index: int, generated_tokens: Sequence[str] = None,
                         phyid_tau: int = 1, phyid_kind: Literal["gaussian", "discrete"] = "gaussian", phyid_redundancy: Literal["MMI", "CCS"] = "MMI",
                         save_dir_path: Union[str, None] = None) -> "PromptPhyID":
        """Create a new PromptPhyID with the given prompt index and model information."""
        if save_dir_path:
            save_file = os.path.join(save_dir_path, f"prompt_{prompt_index}.pkl")
            
            # Check if the file already exists
            if os.path.exists(save_file):
                print(f"PromptPhyID already exists at {save_file}. Loading existing object.")
                return cls.load(save_file)

        obj = cls(prompt_index, model_info, generated_tokens=generated_tokens)
        obj._compute_phyid(prompt_time_series, model_info, phyid_tau=phyid_tau, phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy)
        if save_dir_path:
            obj.save(save_file)
            print(f"PromptPhyID saved to {save_file}")
        return obj

    def _compute_phyid(self, prompt_time_series: PromptTimeSeries, model_info: ModelInformation, phyid_tau: int = 1, 
                       phyid_kind: Literal["gaussian", "discrete"] = "gaussian", phyid_redundancy: Literal["MMI", "CCS"] = "MMI") -> None:
        """Compute the phyid time-series for each node in the prompt time-series."""

        nodes = [(layer_index, node_index) for layer_index, layer in prompt_time_series.layers.items() for node_index in layer.nodes.keys()]
        total_nodes = len(nodes)
        total_pairs = total_nodes * (total_nodes - 1)  # Exclude self-pairs

        samples_seen = 0
        cumulative_time = 0.0

        # Iterate over all pairs of nodes
        for source_layer_index, source_layer_time_series in prompt_time_series.layers.items():
            for source_node_index, source_node_time_series in source_layer_time_series.nodes.items():
                for target_layer_index, target_layer_time_series in prompt_time_series.layers.items():
                    for target_node_index, target_node_time_series in target_layer_time_series.nodes.items():
                        if (source_layer_index == target_layer_index and source_node_index == target_node_index):
                            continue

                        # Compute the PhiID for the pair of nodes and record the time taken
                        t0 = time.perf_counter()
                        phyid_ts = PhyIDTimeSeries.from_time_series(
                            model_info,
                            source_layer_index, source_node_index, target_layer_index, target_node_index,
                            source_time_series=source_node_time_series, target_time_series=target_node_time_series,
                            phyid_tau=phyid_tau, phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy
                        )
                        dt = time.perf_counter() - t0
                        cumulative_time += dt
                        samples_seen += 1
                        
                        # Report progress at logarithmic intervals
                        if samples_seen in {1, 10, 100, 1000, 10000, 100000, 1000000} or samples_seen == total_pairs:
                            avg_time = cumulative_time / samples_seen
                            eta_seconds = avg_time * (total_pairs - samples_seen)
                            eta = timedelta(seconds=int(eta_seconds))
                            print(f"[ETA] {samples_seen}/{total_pairs} done | avg={avg_time:.3f}s | ETA ≈ {eta}", flush=True)
                            # flush the output to ensure it appears immediately

                        # Store result
                        self.phyid[(source_layer_index, source_node_index, target_layer_index, target_node_index)] = phyid_ts

    def compute_extra_atoms(self) -> None:
        """Compute additional atoms for all PhyIDTimeSeries in this prompt."""
        for phyid_ts in self.phyid.values():
            phyid_ts.compute_extra_atoms()
    
    def get_atoms_names(self) -> List[str]:
        """Return the names of the atoms in all PhyIDTimeSeries of this prompt."""
        return self.phyid[next(iter(self.phyid))].get_atoms_names()

    def build_data_array(self) -> xr.DataArray:
        """Stack *all* Φ‑ID atoms into a 6‑D ``xarray.DataArray``.

        Dimensions: ``[atom, source_layer, source_node, target_layer, target_node, time]``.
        The array is cached in ``self.data_array`` and returned.
        """

        if not self.phyid:
            raise RuntimeError("No Φ‑ID data found; call _compute_phyid first.")
        
        atoms = self.get_atoms_names()

        # Enumerate coordinate values
        source_layers = sorted({k[0] for k in self.phyid})
        source_nodes = sorted({k[1] for k in self.phyid})
        target_layers = sorted({k[2] for k in self.phyid})
        target_nodes = sorted({k[3] for k in self.phyid})
        time_len = next(iter(self.phyid.values())).sts.size

        data = np.empty((len(atoms), len(source_layers), len(source_nodes), len(target_layers), len(target_nodes), time_len), dtype=np.float32)

        for (sl, sn, tl, tn), ts in self.phyid.items():
            sL = source_layers.index(sl)
            sN = source_nodes.index(sn)
            tL = target_layers.index(tl)
            tN = target_nodes.index(tn)
            for a_idx, atom in enumerate(atoms):
                data[a_idx, sL, sN, tL, tN, :] = getattr(ts, atom)

        self.data_array = xr.DataArray(
            data,
            dims=["atom", "source_layer", "source_node", "target_layer", "target_node", "time",],
            coords={
                "atom": atoms,
                "source_layer": source_layers,
                "source_node": source_nodes,
                "target_layer": target_layers,
                "target_node": target_nodes,
                "time": np.arange(time_len),
            },
            name="phiid",
            attrs=dict(model=str(self.model_info.model_name)),
        )
        return self.data_array

    @cached_property
    def syn_minus_red_rank(self) -> np.ndarray:
        """Ranks sources by average (sts - rtr), aggregated over target and time."""

        if self.data_array is None:
            self.build_data_array()

        # Step 1: Stack into flat source/target coordinates
        da = self.data_array.stack(
            target=("target_layer", "target_node"),
        )

        # Step 2: Select atoms
        sts = da.sel(atom="sts")  # shape: (source, target, time)
        rtr = da.sel(atom="rtr")  # same shape

        # Step 3: Compute difference and aggregate over target and time
        diff = sts - rtr  # shape: (source, target, time)
        avg_per_source = diff.mean(dim=["source_node", "target", "time"])  # shape: (source,)

        # Step 4: Rank sources in descending order of (sts - rtr)
        rank = avg_per_source.argsort()[::-1].values  # np.ndarray of ranked indices

        return rank

    @cached_property
    def syn_rank_minus_red_rank(self) -> np.ndarray:
        """Ranks sources by average (sts - rtr), aggregated over target and time."""

        if self.data_array is None:
            self.build_data_array()

        # Step 1: Stack into flat source/target coordinates
        da = self.data_array.stack(
            target=("target_layer", "target_node"),
        )

        # Step 2: Select atoms
        sts = da.sel(atom="sts")  # shape: (source, target, time)
        rtr = da.sel(atom="rtr")  # same shape

        # Step 3: Compute difference and aggregate over target and time
        sts_ranked = sts.mean(dim=["source_node", "target", "time"])  # shape: (source,)
        sts_ranked = sts_ranked.argsort()[::-1].values  # np.ndarray of ranked indices

        rtr_ranked = rtr.mean(dim=["source_node", "target", "time"])  # shape: (source,)
        rtr_ranked = rtr_ranked.argsort()[::-1].values  # np.ndarray of ranked indices

        # Step 3: Compute difference and aggregate over target and time
        rank = sts_ranked - rtr_ranked  # shape: (source,)

        return rank

    def plot_syn_minus_red_rank(self, atom="syn_minus_red_rank", plot_dir: Union[str, None] = None) -> None:
        """Plot the rank of sources by average (sts - rtr)."""
        if self.data_array is None:
            self.build_data_array()

        rank = getattr(self, atom)
        plt.figure(figsize=(10, 6))
        plt.plot(rank, marker='o', linestyle='-', color='blue')
        plt.title("Source Nodes Ranked by Average (STS - RTR)")
        plt.xlabel("Source Node Index")
        plt.ylabel("Rank")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if plot_dir:
            plot_dir = os.path.join(plot_dir, "syn_minus_red_rank")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, "rank_plot.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Convenience reductions & plots
    # ------------------------------------------------------------------

    def plot_mean_along(self, atom: str = "sts", varying_dim: str = "time", plot_dir: Union[str, None] = None) -> None:
        """
        Plot the mean of a specific Φ-ID atom along a chosen dimension,
        aggregating over all others.

        Parameters
        ----------
        atom : str
            The Φ-ID atom to select, e.g., 'sts'.
        varying_dim : str
            The dimension along which to plot (e.g., 'time', 'source_layer', etc.).
        """
        if self.data_array is None:
            self.build_data_array()

        if varying_dim not in self.data_array.dims:
            raise ValueError(f"Invalid dimension '{varying_dim}'. Must be one of: {list(self.data_array.dims)}")

        # Compute mean over all dims except the one we want to vary along
        dims_to_reduce = [d for d in self.data_array.dims if d not in ("atom", varying_dim)]
        series = self.data_array.sel(atom=atom).mean(dim=dims_to_reduce)

        series.plot.line(marker="o")
        plt.title(f"Mean {atom.upper()} vs {varying_dim}")
        plt.xlabel(varying_dim.replace('_', ' ').title())
        plt.ylabel(atom)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        if plot_dir:
            plot_dir = os.path.join(plot_dir, f"plot_mean_along/{atom}")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, f"{varying_dim}.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()


    def node_heatmap(self, atom: str = "sts", plot_dir: Union[str, None] = None) -> None:
        """Heat‑map of *atom* averaged over time (source×target)."""
        if self.data_array is None:
            self.build_data_array()

        a = (
            self.data_array.sel(atom=atom)
            .mean(dim="time")
            .stack(source=("source_layer", "source_node"))
            .stack(target=("target_layer", "target_node"))
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(a, cmap="viridis")
        plt.title(f"Mean {atom.upper()} information flow (source → target)")
        plt.xlabel("Target node")
        plt.ylabel("Source node")
        plt.tight_layout()
        if plot_dir:
            plot_dir = os.path.join(plot_dir, f"node_heatmap/{atom}")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, "heatmap.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Heatmap saved to {save_file}")
        else:
            plt.show()

    def save(self, file_path: str) -> None:
        """Save the PromptPhyID object to a pickle file within the specified directory."""
        dir_path = os.path.dirname(file_path)
        try:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"PromptPhyID successfully saved to '{file_path}'.")
        except Exception as e:
            print(f"Error while saving PromptPhyID to '{dir_path}': {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> "PhyIDTimeSeries":
        """Load a PromptPhyID object from a pickle file."""
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            print(f"PromptPhyID successfully loaded from '{file_path}'.")
            return obj
        except Exception as e:
            print(f"Error while loading PromptPhyID from '{file_path}': {e}")
            raise

    def save_data_array(self, file_path: str, *, compression_level: int = 5) -> None:
        """
        Persist the Φ-ID 6-D DataArray to NetCDF format.

        Parameters
        ----------
        file_path : str
            Destination .nc path.
        compression_level : int, default 5
            zlib compression level (0-9).
        """
        da = self.data_array if self.data_array is not None else self.build_data_array()

        # 2 -- Store ModelInformation in attrs
        da.attrs["model_info_json"] = json.dumps(self.model_info.__dict__)

        # 3 -- Write chunk-wise
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        DATA_VAR = da.name or "phiid"
        encoding = {DATA_VAR: dict(zlib=True, complevel=compression_level)}

        # compute=True → stream one chunk at a time
        da.to_netcdf(path, engine="netcdf4", encoding=encoding)
        # da.to_netcdf(path, engine="netcdf4", encoding=encoding, compute=True)
        print(f"Φ-ID DataArray saved → {path}")

    @classmethod
    def load_from_data_array(cls, file_path: str) -> "PhyIDTimeSeries":
        """
        Load a chunked NetCDF produced by `save_data_array`.
        Keeps the DataArray lazy (one-prompt chunks).
        """
        # 1 -- Open lazily; let xarray/dask respect on-disk chunking
        da = xr.open_dataarray(file_path, chunks={"prompt": 1})

        # 2 -- Restore ModelInformation
        if "model_info_json" not in da.attrs:
            raise ValueError("model_info_json attribute missing from file.")
        model_info = ModelInformation.from_dict(json.loads(da.attrs["model_info_json"]))

        # 3 -- Wrap in lightweight MultiPromptPhyID
        obj = cls(model_info)
        obj.data_array = da
        return obj