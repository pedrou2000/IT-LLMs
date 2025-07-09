import sys, os
from hydra import compose, initialize
from pathlib import Path
from omegaconf import OmegaConf

initialize(config_path="../config", version_base="1.3")
cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))

sys.path.insert(0, str(cfg.paths.project_root))

from src.activation_recorder import MultiPromptActivations

print("Loading activations...", flush=True)
data_activations_file = cfg.paths.data_activations_file
activations = MultiPromptActivations.load(file_path=data_activations_file)


from src.time_series_activations import MultiPromptTimeSeries

print("Creating time series from activations...", flush=True)
time_series = MultiPromptTimeSeries.from_activations(
    activations, 
    node_type=cfg.time_series.node_type,
    node_activation=cfg.time_series.node_activation,
    projection_method=cfg.time_series.projection_method, 
    exclude_shared_expert_moe=cfg.time_series.exclude_shared_expert_moe, 
)
time_series.plot(token_x=True, ticks_all_layers=True, plot_dir=cfg.paths.plot_time_series_dir)

from src.phyid_decomposition import MultiPromptPhyID, PromptPhyID, PhyIDTimeSeries

print("Creating phyid decomposition from time series...", flush=True)
data_phyid_file = cfg.paths.data_phyid_file
phyid_comp = MultiPromptPhyID.from_time_series_parallel(
    time_series,
    cfg.phyid.tau,
    cfg.phyid.kind,
    cfg.phyid.redundancy
) 
phyid_comp.save(file_path=data_phyid_file)