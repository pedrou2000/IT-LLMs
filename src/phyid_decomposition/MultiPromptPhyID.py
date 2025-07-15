from __future__ import annotations
""" This module provides the implementation of the PhiID decomposition for time-series data in a multi-prompt setting. """

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union, Tuple
import xarray as xr
import os, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import multiprocessing
import json, pathlib
import pandas as pd
import numpy as np



from src.utils import ModelInformation
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition.PhyIDTimeSeries import PhyIDTimeSeries
from src.phyid_decomposition.PromptPhyID import PromptPhyID


@dataclass
class MultiPromptPhyID:
    """Top‑level container mapping each prompt to a ``PromptTimeSeries``."""

    model_info: ModelInformation
    prompts: Dict[int, PromptPhyID] = field(default_factory=dict, init=False)
    average_prompt_phyid: Union[PromptPhyID, None] = field(default=None, init=False, repr=False)
    data_array: Union[xr.DataArray, None] = field(default=None, init=False, repr=False)


    @classmethod
    def from_time_series(
        cls,
        multi_prompt_time_series: MultiPromptTimeSeries,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",
        save_dir_path: str | None = None,
    ) -> "MultiPromptPhyID":
        """Create a new ``MultiPromptPhyID`` from a ``MultiPromptTimeSeries``."""
        model_info = multi_prompt_time_series.model_info
        obj = cls(model_info)

        for prompt_index, prompt_ts in multi_prompt_time_series.prompts.items():
            # Create a PromptPhyID for each prompt
            print(f"Processing prompt {prompt_index+1}/{len(multi_prompt_time_series.prompts)} with {len(prompt_ts.generated_tokens)} generated tokens.")
            generated_tokens = prompt_ts.generated_tokens
            prompt_phi_id = PromptPhyID.from_time_series(prompt_ts, model_info, prompt_index, generated_tokens, phyid_tau=phyid_tau,
                                                          phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy, save_dir_path=save_dir_path)
            obj.prompts[prompt_index] = prompt_phi_id

        return obj

    @staticmethod
    def _build_one_prompt(args):
        idx, prompt_ts, model_info, phyid_tau, phyid_kind, phyid_redundancy = args
        generated_tokens = prompt_ts.generated_tokens
        phy = PromptPhyID.from_time_series(       # -- heavy work
            prompt_ts, model_info, idx, generated_tokens,
            phyid_tau=phyid_tau,
            phyid_kind=phyid_kind,
            phyid_redundancy=phyid_redundancy,
        )
        return idx, phy

    @classmethod
    def from_time_series_parallel(
        cls,
        multi_prompt_time_series: MultiPromptTimeSeries,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",
        n_workers: int | None = None,     
    ) -> "MultiPromptPhyID":
        """
        Parallel version.  Set ``n_workers`` to the number of CPU cores you
        want to devote (default = all available).
        """
        model_info = multi_prompt_time_series.model_info
        obj = cls(model_info)

        # ---------- pack work ----------
        tasks = [
            (idx, ts, model_info, phyid_tau, phyid_kind, phyid_redundancy)
            for idx, ts in multi_prompt_time_series.prompts.items()
        ]

        # ---------- launch pool ----------
        if n_workers is None:
            n_workers = os.cpu_count() or 1
            print(f"Using all {n_workers} CPU cores for parallel processing.", flush=True)

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(cls._build_one_prompt, t) for t in tasks]

            iterable = as_completed(futures)
            iterable = tqdm(iterable, total=len(futures), desc="Phy-ID")

            for fut in iterable:
                idx, phyid = fut.result()            # propagate exceptions here
                obj.prompts[idx] = phyid

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
    
    def build_data_array(self) -> xr.DataArray:
        """
        Stack per-prompt Φ-ID DataArrays into one 7-D array.

        Output dims:
            [prompt, atom, source_layer, source_node, target_layer, target_node, time]
        """
        # 1. Build (or fetch) each prompt-level DataArray
        da_list, prompt_labels = [], []
        for p_idx, prompt in self.prompts.items():
            da = prompt.build_data_array()        # <-- reuse!
            da_list.append(da)
            prompt_labels.append(p_idx)

        # 2. Concatenate along a new 'prompt' dimension
        big = xr.concat(da_list, dim=pd.Index(prompt_labels, name="prompt"))

        # 3. Attach model-level attrs / encoding as needed
        big.attrs.update(model=str(self.model_info.model_name))
        self.data_array = big

        return big

    def compute_average_prompt_phyid(self) -> PromptPhyID:
        """
        Build (if necessary) the 7-D DataArray, then produce a PromptPhyID
        whose Φ-ID time-series are the mean over prompts, **without**
        collapsing node-pair or time dimensions.
        """
        da = self.data_array if self.data_array is not None else self.build_data_array()
        avg_da = da.mean(dim="prompt") # [atom, source_layer, …, time]

        # ------------------------------------------------------------------
        # 2 · Convert the 6-D DataArray back into a PromptPhyID wrapper
        # ------------------------------------------------------------------
        out = PromptPhyID(
            prompt_index=-1,              # “synthetic” prompt
            model_info=self.model_info,
            generated_tokens=[],          # no natural token stream
        )

        # Enumerate every node-pair coordinate once
        for sl in avg_da.coords["source_layer"].values:
            print(f"Processing source layer {sl}...")  # Debug output
            for sn in avg_da.coords["source_node"].values:
                for tl in avg_da.coords["target_layer"].values:
                    for tn in avg_da.coords["target_node"].values:

                        # Slice all atoms for this pair   (shape ⇒ [atom, time])
                        pair_ts = avg_da.sel(
                            source_layer=sl, source_node=sn,
                            target_layer=tl, target_node=tn
                        )

                        # Build a PhyIDTimeSeries and shove the values in
                        phy_ts = PhyIDTimeSeries(
                            model_info=self.model_info,
                            source_layer_index=int(sl),
                            source_node_index=int(sn),
                            target_layer_index=int(tl),
                            target_node_index=int(tn),
                        )
                        # Each atom lives in pair_ts as pair_ts.sel(atom=atom_name)
                        for atom in pair_ts.coords["atom"].values:
                            setattr(phy_ts, atom, pair_ts.sel(atom=atom).values)

                        # Store
                        out.phyid[(int(sl), int(sn), int(tl), int(tn))] = phy_ts

        self.average_prompt_phyid = out
        return out

    def save(self, dir_path: str) -> None:
        """Save the MultiPromptPhyID object by saving each PromptPhyID to a file."""
        for p_idx, prompt in self.prompts.items():
            file_path = os.path.join(dir_path, f"prompt_{p_idx:03d}.pkl")
            prompt.save(file_path)
        print(f"MultiPromptPhyID successfully saved to directory '{dir_path}'.")
    
    @classmethod
    def load(cls, dir_path: str) -> "MultiPromptPhyID":
        """Load a MultiPromptPhyID object from the PromptPhyID files in the specified directory."""
        prompts = {}
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".pkl"):
                file_path = os.path.join(dir_path, file_name)
                prompt_index = int(file_name.split("_")[1].split(".")[0])
                prompt = PromptPhyID.load(file_path)
                prompts[prompt_index] = prompt
        
        # Create a MultiPromptPhyID object
        model_info = prompts[0].model_info if prompts else ModelInformation()
        obj = cls(model_info)
        obj.prompts = prompts

        print(f"MultiPromptPhyID loaded from directory: {dir_path}")
        return obj
    
    def save_data_array(self, dir_path: str, *, compression_level: int = 5) -> None:
        """
        Persist the Φ-ID 7-D DataArray to NetCDF with **one prompt per chunk**.
        Optionally add further chunk specs via `extra_chunks`.

        Parameters
        ----------
        dir_path : str
            Destination .nc path.
        compression_level : int, default 5
            zlib compression level (0-9).
        """
        da = self.data_array if self.data_array is not None else self.build_data_array()

        for p_idx, prompt in self.prompts.items():
            file_path = os.path.join(dir_path, f"prompt_{p_idx:03d}.nc")
            prompt.save_data_array(file_path, compression_level=compression_level)
        print(f"MultiPromptPhyID DataArray saved to directory: {dir_path}")

    @classmethod
    def load_from_data_array(cls, dir_path: str) -> "MultiPromptPhyID":
        """
        Load a chunked NetCDF produced by `save_data_array`.
        """
        # 1 -- Iterate over all files in the directory
        prompts = {}
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".nc"):
                file_path = os.path.join(dir_path, file_name)
                prompt_index = int(file_name.split("_")[1].split(".")[0])
                prompt = PromptPhyID.load_from_data_array(file_path)
                prompts[prompt_index] = prompt
        # 2 -- Create a MultiPromptPhyID object
        model_info = prompts[0].model_info if prompts else ModelInformation()
        obj = cls(model_info)

        # 3 -- Create the data_array by concatenating all prompts into a single DataArray
        da_list = [prompt.data_array for prompt in prompts.values()]
        obj.data_array = xr.concat(da_list, dim=pd.Index(list(prompts.keys()), name="prompt"))

        print(f"MultiPromptPhyID loaded from directory: {dir_path}")
        return obj