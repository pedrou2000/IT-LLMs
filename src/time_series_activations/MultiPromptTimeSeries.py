"""Time‑series utilities for analysing per‑node activations across multiple prompts."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union
import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils import ModelInformation
from src.activation_recorder import MultiPromptActivations


Projection = Literal["norm", "mean"]  # supported projection options


@dataclass
class NodeTimeSeries:
    """Stores the scalar time‑series for a single node (e.g. an attention head)."""

    node_index: int
    model_info: ModelInformation
    _buffer: List[float] = field(default_factory=list, init=False, repr=False)

    def add_timestep(self, value: float) -> None:  # O(1)
        """Append a new scalar value (cheap list append)."""
        self._buffer.append(float(value))
    
    @property
    def time_series(self) -> np.ndarray:
        """Return the time_series as a *materialised* float32 array (lazy conversion)."""
        if isinstance(self._buffer, list):  # first access ➜ convert & cache
            self._buffer = np.asarray(self._buffer, dtype=np.float32)
        return self._buffer  # type: ignore[return-value]


@dataclass
class LayerTimeSeries:
    """Aggregates ``NodeTimeSeries`` for every node in a layer."""

    layer_index: int
    model_info: ModelInformation
    nodes: Dict[int, NodeTimeSeries] = field(default_factory=dict, init=False)

    def get_or_create_node(self, node_index: int) -> NodeTimeSeries:
        series = self.nodes.get(node_index)
        if series is None:
            series = self.nodes[node_index] = NodeTimeSeries(node_index, self.model_info)
        return series


@dataclass
class PromptTimeSeries:
    """Holds per‑layer time‑series for a single prompt."""

    prompt_index: int
    model_info: ModelInformation
    layers: Dict[int, LayerTimeSeries] = field(default_factory=dict, init=False)
    generated_tokens: Sequence[str] = field(default_factory=list, init=False)

    def get_or_create_layer(self, layer_index: int) -> LayerTimeSeries:
        ts = self.layers.get(layer_index)
        if ts is None:
            ts = self.layers[layer_index] = LayerTimeSeries(layer_index, self.model_info)
        return ts


@dataclass
class MultiPromptTimeSeries:
    """Top‑level container mapping each prompt to a ``PromptTimeSeries``."""

    model_info: ModelInformation
    prompts: Dict[int, PromptTimeSeries] = field(default_factory=dict, init=False)

    @classmethod
    def from_activations(
        cls,
        activations: MultiPromptActivations,
        *,
        node_type: str,
        node_activation: str,
        projection_method: Projection = "norm",
        exclude_shared_expert_moe: bool = False,
    ) -> "MultiPromptTimeSeries":
        """Build and populate a ``MultiPromptTimeSeries`` from recorded activations.

        Parameters
        ----------
        activations : MultiPromptActivations
            Hierarchical activations container (prompts ➜ steps ➜ layers ➜ nodes).
        node_type : str
            Attribute name inside ``layer`` objects (e.g. ``"attention_head"``).
        node_activation : str
            Attribute name inside ``node`` objects whose tensor we will project.
        projection_method : {"norm", "mean"}
            Scalar projection applied to the tensor. Further methods can be added
            by extending the ``_project`` helper.
        """
        obj = cls(activations.model_info)
        obj._populate(activations, node_type, node_activation, projection_method, 
                      exclude_shared_expert_moe=exclude_shared_expert_moe)
        return obj

    def _populate(
        self,
        activations: MultiPromptActivations,
        node_type: str, # e.g. "attention", "moe", "mlp"
        node_activation: str, # e.g. "query", "attention_weights", "attention_outputs", "projected_outputs", "gate_value", "mlp_output", "expert_output"
        projection_method: Projection,
        exclude_shared_expert_moe: bool = False,
    ) -> None:
        """Traverse the activation tree and fill our data‑structure in‑place."""

        for prompt_id, prompt_acts in activations.prompts.items():
            prompt_ts = self._get_or_create_prompt(prompt_id)
            generated_tokens = prompt_acts.generated_tokens

            for step_index, step_acts in prompt_acts.steps.items():
                del step_index  # step granularity handled implicitly by append order

                for layer_index, layer_acts in step_acts.layers.items():
                    layer_ts = prompt_ts.get_or_create_layer(layer_index)

                    node_layer = getattr(layer_acts, node_type, None)
                    if layer_index == 0 and node_type == "moe":
                        continue # skip MoE layer at index 0 as it is an MLP layer
                    
                    if node_layer is None:
                        raise AttributeError(f"Layer {layer_index} lacks node type '{node_type}' for prompt {prompt_id}.")

                    nodes = getattr(node_layer, "nodes", None)
                    if nodes is None:
                        raise AttributeError(f"No 'nodes' attribute in {node_type} of layer {layer_index} (prompt {prompt_id}).")

                    for node_index, node_acts in nodes.items():
                        if hasattr(node_acts, "is_shared") and exclude_shared_expert_moe and node_acts.is_shared:
                            # Skip shared expert nodes in MoE layers if requested
                            continue
                        node_ts = layer_ts.get_or_create_node(node_index)
                        activation = getattr(node_acts, node_activation, None)
                        if node_type == "moe" and activation is None:
                            value = 0.0  # MoE nodes without activation are inactive
                        elif activation is None:
                                raise AttributeError(f"Node {node_index} in layer {layer_index} lacks activation '{node_activation}'.")
                        else:
                            value = self._project(activation, projection_method)
                        node_ts.add_timestep(value)
            # Set the tokens for the prompt time-series
            prompt_ts.generated_tokens = generated_tokens if generated_tokens is not None else []

    def _get_or_create_prompt(self, prompt_index: int, generated_tokens: Sequence[str] = None) -> PromptTimeSeries:
        """Retrieve or create a ``PromptTimeSeries`` for the given prompt index."""
        ts = self.prompts.get(prompt_index)
        if ts is None:
            ts = self.prompts[prompt_index] = PromptTimeSeries(prompt_index, self.model_info)

        ts.generated_tokens = generated_tokens if generated_tokens is not None else []
        return ts

    @staticmethod
    def _project(tensor, method: Projection) -> float:  # noqa: ANN001 – tensor type is backend‑dependent
        """Project the incoming tensor to a scalar."""
        if method == "norm":
            return float(tensor.norm())
        if method == "mean":
            return float(tensor.mean())
        # The ``Literal`` type ensures we never reach here at type‑check time.
        raise ValueError(f"Unsupported projection method: {method}")


    def plot(
        self,
        *,
        token_x: bool | str = "auto",
        figsize_per_layer: float = 2.5,
        ticks_all_layers: bool = False,
        plot_dir: Union[str, None] = None,
    ) -> None:
        """Plot each node’s series (figure‑per‑prompt, subplot‑per‑layer).

        Parameters
        ----------
        token_x : bool | {'auto', True, False}
            * ``True``  – always use generated tokens as the x‑axis labels.
            * ``False`` – always use numeric indices (legacy behaviour).
            * ``'auto'`` – (default) use tokens *only* when they are present.
        figsize_per_layer : float
            Height in inches allocated to each layer subplot.
        show : bool
            Call ``plt.show()`` automatically.
        """
        for prompt_id, prompt_ts in self.prompts.items():
            # decide whether to show tokens on x‑axis
            labels = list(prompt_ts.generated_tokens.values())
            print(f"Prompt {prompt_id} has {len(labels)} generated tokens: {labels}")
            use_tokens = (token_x is True) or (token_x == "auto" and labels)

            layer_ids = sorted(prompt_ts.layers)
            if not layer_ids:
                continue

            fig, axes = plt.subplots(
                len(layer_ids), 1, sharex=True,
                figsize=(10, figsize_per_layer * len(layer_ids)),
            )
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

            # draw each layer
            for ax, layer_id in zip(axes, layer_ids):
                layer_ts = prompt_ts.layers[layer_id]
                for node_id, node_ts in layer_ts.nodes.items():
                    label = f"N{node_id}"
                    ax.plot(range(len(node_ts.time_series)), node_ts.time_series, label=label)
                ax.set_ylabel(f"Layer {layer_id}")
                ax.margins(x=0)

                if ticks_all_layers:
                    for ax in axes:
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=0, fontsize="small")
                        ax.tick_params(labelbottom=True)  # 👈 force showing labels

            # configure shared x‑axis
            if use_tokens:
                axes[-1].set_xticks(range(len(labels)))
                axes[-1].set_xticklabels(labels, rotation=0, fontsize="small")
                axes[-1].set_xlabel("Token")
            else:
                axes[-1].set_xlabel("Timestep")


            handles_by_label: dict[str, matplotlib.artist.Artist] = {}

            for ax in axes:                                   # every subplot
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l and l not in handles_by_label:       # first time we see this label
                        handles_by_label[l] = h               # remember its handle

            if handles_by_label:                              # create a *deduplicated* legend
                fig.legend(
                    handles_by_label.values(), handles_by_label.keys(),
                    fontsize="small", ncol=12,
                    loc="upper right", bbox_to_anchor=(1, 1)
                )


            fig.tight_layout(rect=[0, 0, 1, 0.98])   # leave 8 % of the height free on top

            if plot_dir:
                plot_dir = f"{plot_dir}/prompt_{prompt_id}"
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir, exist_ok=True)
                save_file = f"{plot_dir}/time_series.png"
                fig.savefig(save_file, dpi=300)
                print(f"Time-series plot saved to {save_file}")
            else:
                plt.show()
