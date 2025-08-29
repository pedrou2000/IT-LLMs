#!/usr/bin/env python
import multiprocessing as mp
import sys, os
from pathlib import Path

def main():
    # ---------------- Hydra config ----------------
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    initialize(config_path="../config", version_base="1.3")
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    sys.path.insert(0, str(cfg.paths.project_root))

    from src.activation_recorder import MultiPromptActivations
    from src.time_series_activations import MultiPromptTimeSeries
    from src.phyid_decomposition import MultiPromptPhyID

    # ---------------- Pipeline ----------------
    data_phyid_path = cfg.paths.data_phyid_dir

    print(f"Loading phyid from {data_phyid_path}", flush=True)
    phyid = MultiPromptPhyID.load(dir_path=data_phyid_path)
    print("Building data array for phyid", flush=True)
    # phyid.build_data_array()

    print("Computing average prompt phyid", flush=True)
    # phyid = phyid.compute_average_prompt_phyid(save_dir_path=data_phyid_path)
    phyid = phyid.compute_average_prompt_phyid_stream(save_dir_path=data_phyid_path)

    average_prompt_phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=data_phyid_path)

    print("Plotting phyid results...", flush=True)
    plot_dir = cfg.paths.plot_phyid_dir
    average_prompt_phyid.node_heatmap(atom='sts', plot_dir=plot_dir)
    average_prompt_phyid.plot_mean_along('sts', varying_dim='source_layer', plot_dir=plot_dir)


# ----------------------------------------------------------------------
if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()