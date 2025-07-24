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

    from src.phyid_decomposition import MultiPromptPhyID, PromptPhyID, PhyIDTimeSeries
    from src.ranked_deactivation_analysis import RankedDeactivationAnalysis, RankedDeactivationResults, RankedDeactivationExperiment

    # ---------------- Load model and tokenizer ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_name,
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()                    # ← puts *every* sub‑module in eval mode
    torch.set_grad_enabled(False)   # optional but avoids autograd bookkeeping
    # model.config._attn_implementation = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_name,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    data_phyid_path = cfg.paths.data_phyid_dir

    # ---------------- Load phyid decomposition ----------------
    print(f"Loading phyid from {data_phyid_path}")
    phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=data_phyid_path)
    node_ranking = phyid.syn_minus_red_rank
    phyid.plot_syn_minus_red_rank_per_node(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_syn_minus_red_rank_per_layer(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    print(f"Synergy–Redundancy Rank of N03-L08: {node_ranking.sel(source_node=3, source_layer=8).values}")
    

    # ---------------- Run ranked deactivation experiment ----------------
    experiment = RankedDeactivationExperiment(
        analysis_kwargs=dict(
            model=model,
            tokenizer=tokenizer,
            prompts=OmegaConf.to_container(cfg.generation.prompts, resolve=True),
            chat_template=cfg.model.apply_chat_template,
            node_ranking=node_ranking,
            max_new_tokens=cfg.generation.max_new_tokens,
        )
    )
    experiment.run_default_and_random(
        deactivate_k_nodes_per_iteration=cfg.deactivation_analysis.deactivate_k_nodes_per_iteration,
        max_deactivated_nodes=cfg.deactivation_analysis.max_deactivated_nodes,
        micro_batch_size=100,
        n_randomised_runs=cfg.deactivation_analysis.n_randomised_runs,
    )
    experiment.plot_overall(plot_dir=cfg.paths.plot_deactivation_dir)
    experiment.plot_per_category(plot_dir=cfg.paths.plot_deactivation_dir)




# ----------------------------------------------------------------------
if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()
