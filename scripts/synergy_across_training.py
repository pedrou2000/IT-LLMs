"""
Batch‑processing script for Pythia checkpoints
---------------------------------------------
Runs *exactly* the same activation‑recording ➜ time‑series ➜ PhiID pipeline you
already tested, but loops over every training snapshot from **step000000** to
**step143000** in 1 k increments.

The script expects your usual Hydra config folder (`../config`) and re‑uses all
paths that depend on `cfg.model.revision`.  If your paths are written with
`${model.revision}` interpolations (recommended), each checkpoint’s results
will automatically land in its own sub‑directory, e.g.:

    outputs/step050000/activations/
    outputs/step050000/phyid/

Adjust the `CHECKPOINT_RANGE` or CLI args as you like.  Heavy job – make sure
you have the storage and GPU availability.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path
from typing import List, Any

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- project‑local imports ----------------------------------------------------
from src.utils import perturb_model
from src.activation_recorder import ActivationRecorder, MultiPromptActivations
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition import MultiPromptPhyID

# -----------------------------------------------------------------------------
# Helper: run the *full* pipeline for one checkpoint
# -----------------------------------------------------------------------------

def run_for_checkpoint(step: int, base_cfg: Any) -> None:
    """Run recording → PhiID for a single training snapshot."""

    # Deep‑copy the OmegaConf so each run is clean
    cfg = OmegaConf.create(base_cfg)

    revision = f"step{step}"
    cfg.model.revision = revision  # <-- this automatically updates paths if
                                   #     they use ${model.revision}
    new_shortcode = f"P-1-{step}"
    cfg.model.shortcode = new_shortcode
    cfg.model.it = f"base-{step}"

    print(f"\n=== Processing checkpoint {revision} ===")

    # ---------------------------------------------------------------------
    # 1)  Load model & tokenizer
    # ---------------------------------------------------------------------
    model_name: str = cfg.model.hf_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    # ---------------------------------------------------------------------
    # 2)  Record activations
    # ---------------------------------------------------------------------
    prompts: List[str] = cfg.generation.prompts
    # flatten [ [..], [..] ] → [..]
    if isinstance(prompts, list) and all(isinstance(p, list) for p in prompts):
        prompts = [item for sublist in prompts for item in sublist]

    recorder = ActivationRecorder(model, tokenizer)
    activations: MultiPromptActivations = recorder.record_prompts(
        prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        prompt_template=cfg.model.apply_chat_template,
    )
    activations.verify_recorded_activations(
        prompts=prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        tokenizer=tokenizer,
        diff_q_size=True,
    )
    activations.save(cfg.paths.data_activations_file)

    # ---------------------------------------------------------------------
    # 3)  Time‑series + PhiID
    # ---------------------------------------------------------------------
    time_series = MultiPromptTimeSeries.from_activations(
        activations,
        node_type=cfg.time_series.node_type,
        node_activation=cfg.time_series.node_activation,
        projection_method=cfg.time_series.projection_method,
        exclude_shared_expert_moe=cfg.time_series.exclude_shared_expert_moe,
    )
    time_series.plot(
        token_x=True,
        ticks_all_layers=True,
        plot_dir=cfg.paths.plot_time_series_dir,
    )

    phyid_comp = MultiPromptPhyID.from_time_series(
        time_series,
        cfg.phyid.tau,
        cfg.phyid.kind,
        cfg.phyid.redundancy,
    )
    phyid_comp.save(dir_path=cfg.paths.data_phyid_dir)

    # optional extra plots / metrics
    data_phyid_path = cfg.paths.data_phyid_dir
    phyid = MultiPromptPhyID.load(dir_path=data_phyid_path)
    print("Building data array for phyid")
    phyid.build_data_array()
    print("Computing average prompt phyid")
    phyid = phyid.compute_average_prompt_phyid(save_dir_path=cfg.paths.data_phyid_dir)

    phyid.plot_mean_along('sts', 'source_layer', plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_mean_along('rtr', 'source_layer', plot_dir=cfg.paths.plot_phyid_dir)
    node_ranking = phyid.syn_minus_red_rank
    phyid.plot_syn_minus_red_rank_per_node(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_syn_minus_red_rank_per_layer(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    print(f"Synergy–Redundancy Rank of N03-L08: {node_ranking.sel(source_node=3, source_layer=8).values}")

    # ---------------------------------------------------------------------
    # 4)  House‑keeping – free GPU RAM
    # ---------------------------------------------------------------------
    del model, recorder, activations, time_series, phyid_comp, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# Main entry point – iterate over the whole training trajectory
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run activation+PhyID pipeline for many Pythia checkpoints")
    parser.add_argument("--start", type=int, default=0, help="First step (inclusive)")
    parser.add_argument("--end",   type=int, default=143000, help="Last step (inclusive)")
    parser.add_argument("--skip",  type=int, default=5000, help="Step size between checkpoints")
    args = parser.parse_args()

    # Load *base* Hydra config once.  All overrides happen in‑memory.
    with initialize(config_path="../config", version_base="1.3"):
        base_cfg = compose(config_name="config")

    # Ensure project root is discoverable
    import sys
    sys.path.insert(0, str(base_cfg.paths.project_root))

    steps = [2**i for i in range(0, 9)] + [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

    # for step in range(args.end, args.start - 1, -args.skip):
    # for step in range(args.start, args.end + 1, args.skip):
    for step in steps:
        try:
            run_for_checkpoint(step, base_cfg)
        except Exception as e:
            print(f"[Warning] Checkpoint step{step:06d} failed with error: {e}")
            continue


if __name__ == "__main__":
    main()
