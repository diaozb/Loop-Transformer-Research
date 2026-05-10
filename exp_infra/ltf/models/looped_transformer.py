from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ltf.config import ModelConfig

from .minimal_gpt2 import GPT2Config, GPT2Model


@dataclass
class LoopedForwardTrace:
    logits: List[torch.Tensor]
    hidden: List[torch.Tensor]


class GeneralLoopedTransformer(nn.Module):
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int = 128,
        n_layer: int = 12,
        n_head: int = 4,
        linear_embedding: bool = False,
        use_wpe: bool = False,
        wpe_mode: Optional[str] = None,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.use_wpe = use_wpe
        self.wpe_mode = wpe_mode if wpe_mode is not None else ("all" if use_wpe else "none")
        if self.wpe_mode not in ("none", "once", "all"):
            raise ValueError(f"Unsupported wpe_mode: {self.wpe_mode}")
        self.use_rope = use_rope
        if self.use_wpe and self.use_rope:
            raise ValueError("use_wpe and use_rope cannot both be True.")
        configuration.use_rope = use_rope
        configuration.rope_theta = rope_theta

        if linear_embedding:
            self._read_in = nn.Linear(n_dims, n_embd)
        else:
            self._read_in = nn.Embedding(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

    def forward_no_position(self, zs: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        embeds = self._read_in(zs)
        output = self._backbone.forward_no_position(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self._read_out(output)

    def _should_add_wpe(self, add_wpe: bool | None = None, step_idx: int | None = None) -> bool:
        if not getattr(self, "use_wpe", False):
            return False
        if add_wpe is not None and not add_wpe:
            return False
        if self.wpe_mode == "none":
            return False
        if self.wpe_mode == "all":
            return True
        if self.wpe_mode == "once":
            return step_idx in (None, 0)
        raise ValueError(f"Unsupported wpe_mode: {self.wpe_mode}")

    def forward_single(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        add_wpe: bool | None = None,
        step_idx: int | None = None,
    ) -> torch.Tensor:
        if self._should_add_wpe(add_wpe=add_wpe, step_idx=step_idx):
            batch_size, seq_len, _ = embeds.shape
            position_ids = torch.arange(seq_len, device=embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_ids = position_ids % self.n_positions
            return self._backbone(
                inputs_embeds=embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        return self._backbone.forward_no_position(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

    def looped_forward(
        self,
        zs: torch.Tensor,
        horizon: int,
        attention_mask: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        return self.collect_looped_forward(zs, horizon, attention_mask).logits

    def collect_looped_forward(
        self,
        zs: torch.Tensor,
        horizon: int,
        attention_mask: torch.Tensor | None = None,
    ) -> LoopedForwardTrace:
        input_embed = self._read_in(zs)
        output = torch.zeros_like(input_embed).to(input_embed.device)
        logits_list: List[torch.Tensor] = []
        hidden_list: List[torch.Tensor] = []
        use_wpe = getattr(self, "use_wpe", False)
        for step in range(horizon):
            output = self.forward_single(
                output + input_embed,
                attention_mask=attention_mask,
                add_wpe=use_wpe,
                step_idx=step,
            )
            hidden_list.append(output)
            logits_list.append(self._read_out(output).clone())
        return LoopedForwardTrace(logits=logits_list, hidden=hidden_list)

    def looped_forward_without(
        self,
        zs: torch.Tensor,
        horizon: int,
        attention_mask: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        output = self._read_in(zs)
        logits_list: List[torch.Tensor] = []
        use_wpe = getattr(self, "use_wpe", False)
        for step in range(horizon):
            output = self.forward_single(
                output,
                attention_mask=attention_mask,
                add_wpe=use_wpe,
                step_idx=step,
            )
            logits_list.append(self._read_out(output).clone())
        return logits_list


def build_looped_model(config: ModelConfig) -> GeneralLoopedTransformer:
    if config.family != "gpt2":
        raise NotImplementedError(f"Unsupported model family: {config.family}")
    return GeneralLoopedTransformer(
        n_dims=config.n_dims,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        linear_embedding=config.linear_embedding,
        use_wpe=config.use_wpe,
        wpe_mode=config.wpe_mode,
        use_rope=config.use_rope,
        rope_theta=config.rope_theta,
    )
