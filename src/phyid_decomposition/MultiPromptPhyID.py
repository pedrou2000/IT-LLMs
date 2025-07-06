""" This module provides the implementation of the PhiID decomposition for time-series data in a multi-prompt setting. """

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union, Tuple
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import os, pickle
import time
from datetime import timedelta

from phyid.calculate import calc_PhiID 

from src.utils import ModelInformation
from src.time_series_activations import MultiPromptTimeSeries, PromptTimeSeries, LayerTimeSeries, NodeTimeSeries


INFORMATION_DYNAMICS = {
    "storage": ["rtr", "xtx", "yty", "sts"],
    "copy": ["xtr", "ytr"],
    "transfer": ["xty", "ytx"],
    "erasure": ["rtx", "rty"],
    "downward_causation": ["stx", "sty", "str_"],
    "upward_causation": ["xts", "yts", "rts"],
    "information_storage": ["rtr", "rtx", "xtr", "xtx"],
    "transfer_entropy_source_past_to_target_future": ["str_", "sty", "xtr", "xty"],
    "causal_density": [(2,"str_"), "sty", "xtr", "xty", "stx", "ytx", "ytr"],
    "integrated_information": ["sts", "xts", "yts", "stx", "sty", "xty", "ytx", "rts", "str_", (-1, "rtr")], 
    "mutual_information": [
        "rtr", "rtx", "rty", "rts", 
        "str_", "stx", "sty", "sts", 
        "xtr", "xtx", "xty", "xts", 
        "ytr", "ytx", "yty", "yts", 
    ]
}


@dataclass
class PhyIDTimeSeries:

    model_info: ModelInformation
    source_layer_index: int
    source_node_index: int
    target_layer_index: int
    target_node_index: int

    xtx: List[float] = field(default_factory=list, repr=False)
    xty: List[float] = field(default_factory=list, repr=False)
    xtr: List[float] = field(default_factory=list, repr=False)
    xts: List[float] = field(default_factory=list, repr=False)

    ytx: List[float] = field(default_factory=list, repr=False)
    yty: List[float] = field(default_factory=list, repr=False)
    ytr: List[float] = field(default_factory=list, repr=False)
    yts: List[float] = field(default_factory=list, repr=False)

    rtx: List[float] = field(default_factory=list, repr=False)
    rty: List[float] = field(default_factory=list, repr=False)
    rtr: List[float] = field(default_factory=list, repr=False)
    rts: List[float] = field(default_factory=list, repr=False)

    stx: List[float] = field(default_factory=list, repr=False)
    sty: List[float] = field(default_factory=list, repr=False)
    str_: List[float] = field(default_factory=list, repr=False)  # 'str' is a built-in, so use 'str_'
    sts: List[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_time_series(
        cls,
        model_info: ModelInformation,
        source_layer_index: int,
        source_node_index: int,
        target_layer_index: int,
        target_node_index: int,
        source_time_series: NodeTimeSeries = None,
        target_time_series: NodeTimeSeries = None,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",
    ) -> "PhyIDTimeSeries":
        """ Compute the phyid decomposition for a given source and target nodes. """
        obj = cls(model_info, source_layer_index, source_node_index, target_layer_index, target_node_index)
        if source_time_series is not None and target_time_series is not None:
            obj._compute_phyid(source_time_series, target_time_series, phyid_tau=phyid_tau, phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy)
        return obj

    def _compute_phyid(
        self,
        source_time_series: NodeTimeSeries,
        target_time_series: NodeTimeSeries,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",    
    ) -> None:
        """Compute the phyid decomposition for the given source and target time-series."""
        atoms_res, calc_res = calc_PhiID(
            src=source_time_series.time_series, 
            trg=target_time_series.time_series, 
            tau=phyid_tau, 
            kind=phyid_kind, 
            redundancy=phyid_redundancy
        )
        self.fill_from_atoms(atoms_res)


    def fill_from_atoms(self, atoms_res: Dict[str, Union[float, np.ndarray]]) -> None:
        """Fill the PhyIDTimeSeries from the atoms of the decomposition."""
        self.xtx = np.asarray(atoms_res["xtx"], dtype=np.float32)
        self.xty = np.asarray(atoms_res["xty"], dtype=np.float32)
        self.xtr = np.asarray(atoms_res["xtr"], dtype=np.float32)
        self.xts = np.asarray(atoms_res["xts"], dtype=np.float32)
        self.ytx = np.asarray(atoms_res["ytx"], dtype=np.float32)
        self.yty = np.asarray(atoms_res["yty"], dtype=np.float32)
        self.ytr = np.asarray(atoms_res["ytr"], dtype=np.float32)
        self.yts = np.asarray(atoms_res["yts"], dtype=np.float32)
        self.rtx = np.asarray(atoms_res["rtx"], dtype=np.float32)
        self.rty = np.asarray(atoms_res["rty"], dtype=np.float32)
        self.rtr = np.asarray(atoms_res["rtr"], dtype=np.float32)
        self.rts = np.asarray(atoms_res["rts"], dtype=np.float32)
        self.stx = np.asarray(atoms_res["stx"], dtype=np.float32)
        self.sty = np.asarray(atoms_res["sty"], dtype=np.float32)
        self.str_ = np.asarray(atoms_res["str_"], dtype=np.float32)
        self.sts = np.asarray(atoms_res["sts"], dtype=np.float32)
    
    def compute_extra_atoms(self) -> None:
        """ Compute additional atoms based on the existing ones. """
        # Information dynamics atoms
        for extra_atom, dependencies in INFORMATION_DYNAMICS.items():
            extra_ts = np.zeros_like(self.xtx, dtype=np.float32)
            for dep in dependencies:
                multiplier = 1 if isinstance(dep, str) else dep[0]
                dep_atom = dep if isinstance(dep, str) else dep[1]
                extra_ts += multiplier * getattr(self, dep_atom)
            setattr(self, extra_atom, extra_ts)
        
        # Mutual informaiton normalized atoms
        mi = self.mutual_information
        for atom in self.get_atoms_names():
            setattr(self, f"{atom}_normalized", getattr(self, atom) / mi)
        

    def get_atoms_names(self) -> List[str]:
        """Return the names of the atoms in this PhyIDTimeSeries."""
        return sorted(k for k, v in vars(self).items() if isinstance(v, (list, np.ndarray)))

@dataclass
class PromptPhyID:
    """Container for the PhiID time-series of a single prompt. """

    prompt_index: int
    model_info: ModelInformation
    generated_tokens: Sequence[str] = field(default_factory=list)
    phyid: Dict[Tuple[int, int, int, int], PhyIDTimeSeries] = field(default_factory=dict, init=False, repr=False) # (source_layer_index, source_node_index, target_layer_index, target_node_index) -> PhyIDTimeSeries

    def get_phyid(self, source_layer_index: int, source_node_index: int, target_layer_index: int, target_node_index: int) -> PhyIDTimeSeries:
        """Retrieve or create a PhyIDTimeSeries for the given indices."""
        key = (source_layer_index, source_node_index, target_layer_index, target_node_index)
        if key not in self.phyid:
            raise KeyError(f"PhyIDTimeSeries for source layer {source_layer_index}, source node {source_node_index}, target layer {target_layer_index}, target node {target_node_index} not found.")
        return self.phyid[key]
    
    @classmethod
    def from_time_series(cls, prompt_time_series: PromptTimeSeries, model_info: ModelInformation, prompt_index: int, generated_tokens: Sequence[str] = None) -> "PromptPhyID":
        """Create a new PromptPhyID with the given prompt index and model information."""
        obj = cls(prompt_index, model_info, generated_tokens=generated_tokens)
        obj._compute_phyid(prompt_time_series, model_info)
        return obj

    def _compute_phyid(self, prompt_time_series: PromptTimeSeries, model_info: ModelInformation) -> None:
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
                            source_layer_index,
                            source_node_index,
                            target_layer_index,
                            target_node_index,
                            source_time_series=source_node_time_series,
                            target_time_series=target_node_time_series
                        )
                        dt = time.perf_counter() - t0
                        cumulative_time += dt
                        samples_seen += 1
                        
                        # Report progress at logarithmic intervals
                        if samples_seen in {1, 10, 100, 1000, 10000, 100000, 1000000} or samples_seen == total_pairs:
                            avg_time = cumulative_time / samples_seen
                            eta_seconds = avg_time * (total_pairs - samples_seen)
                            eta = timedelta(seconds=int(eta_seconds))
                            print(f"[ETA] {samples_seen}/{total_pairs} done | avg={avg_time:.3f}s | ETA ≈ {eta}")

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

    # ------------------------------------------------------------------
    # Convenience reductions & plots
    # ------------------------------------------------------------------

    def plot_mean_along(self, atom: str = "sts", varying_dim: str = "time") -> None:
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
        plt.show()

    def node_heatmap(self, atom: str = "sts") -> None:
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
        plt.show()




@dataclass
class MultiPromptPhyID:
    """Top‑level container mapping each prompt to a ``PromptTimeSeries``."""

    model_info: ModelInformation
    prompts: Dict[int, PromptPhyID] = field(default_factory=dict, init=False)

    @classmethod
    def from_time_series(
        cls,
        multi_prompt_time_series: MultiPromptTimeSeries,
    ) -> "MultiPromptPhyID":
        """Create a new ``MultiPromptPhyID`` from a ``MultiPromptTimeSeries``."""
        model_info = multi_prompt_time_series.model_info
        obj = cls(model_info)

        for prompt_index, prompt_ts in multi_prompt_time_series.prompts.items():
            # Create a PromptPhyID for each prompt
            print(f"Processing prompt {prompt_index+1}/{len(multi_prompt_time_series.prompts)} with {len(prompt_ts.generated_tokens)} generated tokens.")
            generated_tokens = prompt_ts.generated_tokens
            prompt_phi_id = PromptPhyID.from_time_series(prompt_ts, model_info, prompt_index, generated_tokens)
            obj.prompts[prompt_index] = prompt_phi_id

        return obj

    def get_prompt(self, prompt_index: int) -> PromptPhyID:
        """Retrieve the PromptPhyID for a given prompt index."""
        if prompt_index not in self.prompts:
            raise KeyError(f"PromptPhyID for prompt index {prompt_index} not found.")
        return self.prompts[prompt_index]
    
    def compute_extra_atoms(self) -> None:
        """Compute additional atoms for all PhyIDTimeSeries in all prompts."""
        for prompt in self.prompts.values():
            prompt.compute_extra_atoms()

    def save(self, dir_path: str) -> None:
        """Save the MultiPromptPhyID object to a pickle file within the specified directory."""
        try:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, "multi_prompt_phyid.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"MultiPromptPhyID successfully saved to '{file_path}'.")
        except Exception as e:
            print(f"Error while saving MultiPromptPhyID to '{dir_path}': {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> "MultiPromptPhyID":
        """Load a MultiPromptPhyID object from a pickle file."""
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            print(f"MultiPromptPhyID successfully loaded from '{file_path}'.")
            return obj
        except Exception as e:
            print(f"Error while loading MultiPromptPhyID from '{file_path}': {e}")
            raise
