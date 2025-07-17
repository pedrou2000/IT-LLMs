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

    data_phyid_dir = cfg.paths.data_phyid_dir

    print(f"Loading phyid from {data_phyid_dir}")
    phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=data_phyid_dir)
    phyid.compute_extra_atoms()
    file_path = os.path.join(data_phyid_dir, "average.pkl")
    phyid.save(file_path=file_path)


# ----------------------------------------------------------------------
if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()
