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
    time_series.plot(token_x=True, ticks_all_layers=True, plot_dir=cfg.paths.plot_time_series_dir)

    print("Creating phyid decomposition from time series...", flush=True)
    phyid = MultiPromptPhyID.from_time_series( 
        time_series,
        phyid_tau        = cfg.phyid.tau,
        phyid_kind       = cfg.phyid.kind,
        phyid_redundancy = cfg.phyid.redundancy,
        # n_workers        = 8,
        save_dir_path = cfg.paths.data_phyid_dir,
        data_array_only = True,  # If True, only compute the data array without saving or creating PromptPhyID objects
        average_time = True,  # If True, compute the average
    )
    phyid.save(dir_path=cfg.paths.data_phyid_dir)
    print("Building data array for phyid")
    phyid.build_data_array()
    print("Computing average prompt phyid")
    phyid = phyid.compute_average_prompt_phyid(save_dir_path=cfg.paths.data_phyid_dir)

    print("Plotting phyid results...", flush=True)
    plot_dir = cfg.paths.plot_phyid_dir
    phyid.node_heatmap(atom='sts', plot_dir=plot_dir)
    phyid.plot_mean_along('sts', varying_dim='source_layer', plot_dir=plot_dir)

    node_ranking = phyid.syn_minus_red_rank
    phyid.plot_syn_minus_red_rank_per_node(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_syn_minus_red_rank_per_layer(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    print(f"Synergy–Redundancy Rank of N03-L08: {node_ranking.sel(source_node=3, source_layer=8).values}")


if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()
