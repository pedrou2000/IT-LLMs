from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
from bct import efficiency_wei, modularity_louvain_und
import xarray as xr
from scipy import sparse
from functools import cached_property

from src.utils import ModelInformation
from src.phyid_decomposition import PromptPhyID
from src.utils import get_layer_node_indeces, get_node_index




@dataclass
class AtomConnectivityGraph:

    model_info: ModelInformation
    atom: str
    graph: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def _dense(self) -> np.ndarray:
        """Return a dense NumPy copy of the adjacency matrix."""
        return self.graph.toarray() if sparse.issparse(self.graph) else self.graph

    @cached_property
    def global_efficiency(self) -> float:
        """ Calculate the global efficiency of the graph. Global efficiency is defined as the 
        average inverse shortest path length between all pairs of nodes. """
        return efficiency_wei(self._dense())

    @cached_property
    def modularity(self) -> float:
        """ Calculate the local efficiency of the graph. Local efficiency is defined as the 
        average global efficiency of the subgraphs formed by removing each node and its incident edges. """
        return modularity_louvain_und(self._dense())[1]
    
    def plot_graph(self, ax=None, title: Optional[str] = None, cmap: str = "viridis", **kwargs) -> None:
        """ Plot the connectivity graph using matplotlib. """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        G = nx.from_numpy_array(self.graph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', cmap=cmap, **kwargs)

        if title:
            ax.set_title(title)
        
        plt.show()


@dataclass
class PromptGraphTheoreticalAnalysis:

    model_info: ModelInformation
    atom_graphs: Dict[str, AtomConnectivityGraph] = field(default_factory=dict)

    @classmethod
    def from_phyid_data(
        cls,
        prompt_phyid: PromptPhyID,
        *,
        use_sparse: bool = True,          # switch to False for dense matrices
        ensure_extra_atoms: bool = True,  # recompute extras only if missing
    ) -> "PromptGraphTheoreticalAnalysis":
        """
        Build one adjacency matrix per Φ-ID atom using the cached 6-D
        ``PromptPhyID.data_array``.  The whole operation is vectorised.

        Parameters
        ----------
        use_sparse
            Store the matrices as CSR sparse objects to save RAM.
        ensure_extra_atoms
            Call ``compute_extra_atoms`` if no *_normalized atoms exist.
        """
        # ── Make sure every atom we want is present ─────────────────────────────
        # if ensure_extra_atoms and not any(a.endswith("_normalized") for a in prompt_phyid.get_atoms_names()):
            # prompt_phyid.compute_extra_atoms()   # adds info-dynamics, normalised…

        # ── Get or build the 6-D DataArray ─────────────────────────────────────
        if prompt_phyid.data_array is None:
            prompt_phyid.build_data_array()

        da: xr.DataArray = prompt_phyid.data_array     # alias

        # 1. collapse the time axis with a vectorised mean
        da_mean = da.mean(dim="time", keepdims=False)  # dims: atom SL SN TL TN

        # 2. stack (layer,node) → flat single index for src & trg
        da_flat = da_mean.stack(
            source=("source_layer", "source_node"),
            target=("target_layer", "target_node"),
        )                                              # dims: atom source target

        atoms = da_flat.coords["atom"].values
        V     = da_flat.sizes["source"]                # |V| = layers·nodes_per_layer
        mi    = prompt_phyid.model_info

        # ── Convert each atom slice to a matrix ────────────────────────────────
        atom_graphs: Dict[str, AtomConnectivityGraph] = {}
        for atom in atoms:
            mat = da_flat.sel(atom=atom).values.astype(np.float32)  # shape (V,V)

            if use_sparse:
                mat = sparse.csr_matrix(mat)         # zero-values are dropped

            atom_graphs[str(atom)] = AtomConnectivityGraph(
                model_info=mi,
                atom=str(atom),
                graph=mat,
            )

        return cls(model_info=mi, atom_graphs=atom_graphs)

    
    def plot_atom_graph(self, atom: str):
        """ Plot the connectivity graph for a specific atom. """
        if atom not in self.atom_graphs:
            raise ValueError(f"Atom '{atom}' not found in the analysis.")
        
        atom_graph = self.atom_graphs[atom]
        atom_graph.plot_graph(title=f"Connectivity Graph for Atom: {atom}", cmap="viridis")
    
    def global_efficiency(self, atom: str) -> float:
        """ Calculate the global efficiency of the graph for a specific atom. """
        if atom not in self.atom_graphs:
            raise ValueError(f"Atom '{atom}' not found in the analysis.")
        
        return self.atom_graphs[atom].global_efficiency

    def modularity(self, atom: str) -> float:
        """ Calculate the local efficiency of the graph for a specific atom. """
        if atom not in self.atom_graphs:
            raise ValueError(f"Atom '{atom}' not found in the analysis.")
        
        return self.atom_graphs[atom].modularity


