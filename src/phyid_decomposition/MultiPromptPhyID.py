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
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI"
    ) -> "MultiPromptPhyID":
        """Create a new ``MultiPromptPhyID`` from a ``MultiPromptTimeSeries``."""
        model_info = multi_prompt_time_series.model_info
        obj = cls(model_info)

        for prompt_index, prompt_ts in multi_prompt_time_series.prompts.items():
            # Create a PromptPhyID for each prompt
            print(f"Processing prompt {prompt_index+1}/{len(multi_prompt_time_series.prompts)} with {len(prompt_ts.generated_tokens)} generated tokens.")
            generated_tokens = prompt_ts.generated_tokens
            prompt_phi_id = PromptPhyID.from_time_series(prompt_ts, model_info, prompt_index, generated_tokens, 
                                                         phyid_tau=phyid_tau, phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy)
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

    def save(self, file_path: str) -> None:
        """Save the MultiPromptPhyID object to a pickle file within the specified directory."""
        dir_path = os.path.dirname(file_path)
        try:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
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
    
    def save_data_array(self, file_path: str, *, compression_level: int = 5,) -> None:
        """ Persist only the Φ-ID DataArray (NetCDF) **including model_info**. """
        da = self.data_array if self.data_array is not None else self.build_data_array()

        # ------------------------------------------------------------------
        # 1 · Serialize `ModelInformation` as JSON and stuff it in .attrs
        # ------------------------------------------------------------------
        mi_dict = self.model_info.__dict__
        da.attrs["model_info_json"] = json.dumps(mi_dict)

        # ------------------------------------------------------------------
        # 2 · Write NetCDF with compression
        # ------------------------------------------------------------------
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        DATA_VAR = da.name if da.name else "phiid"
        encoding = {DATA_VAR: dict(zlib=True, complevel=compression_level)}

        da.to_netcdf(path, encoding=encoding, engine="netcdf4")
        print(f"Φ-ID DataArray + model metadata saved → {path}")

    @classmethod
    def load_from_data_array(cls, file_path: str) -> "MultiPromptPhyID":
        """
        Recreate a thin MultiPromptPhyID wrapper, restoring `model_info`
        from the JSON stored in the NetCDF file.
        """
        da = xr.open_dataarray(file_path)

        # ----- rebuild ModelInformation -----
        if "model_info_json" not in da.attrs:
            raise ValueError("model_info_json attribute missing from file.")

        mi_dict = json.loads(da.attrs["model_info_json"])
        model_info = ModelInformation.from_dict(mi_dict)

        # ----- return a lightweight wrapper -----
        obj = cls(model_info)
        obj.data_array = da
        return obj
