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
    print("Loading activations...", flush=True)
    activations = MultiPromptActivations.load(file_path=cfg.paths.data_activations_file)

    print("Creating time series from activations...", flush=True)
    time_series = MultiPromptTimeSeries.from_activations(
        activations,
        node_type           = cfg.time_series.node_type,
        node_activation     = cfg.time_series.node_activation,
        projection_method   = cfg.time_series.projection_method,
        exclude_shared_expert_moe = cfg.time_series.exclude_shared_expert_moe,
    )
    time_series.plot(token_x=True, ticks_all_layers=True,
                     plot_dir=cfg.paths.plot_time_series_dir)

    print("Creating phyid decomposition from time series...", flush=True)
    phyid_comp = MultiPromptPhyID.from_time_series_parallel( 
        time_series,
        phyid_tau        = cfg.phyid.tau,
        phyid_kind       = cfg.phyid.kind,
        phyid_redundancy = cfg.phyid.redundancy,
    )
    phyid_comp.save(file_path=cfg.paths.data_phyid_file)


# ----------------------------------------------------------------------
if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()
