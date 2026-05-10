import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

def build_general_model(conf, y_dim=1):
    if conf.family == "gpt2":
        model = GeneralTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            linear_embedding=conf.linear_embedding,
            use_wpe=conf.use_wpe,
            wpe_mode=getattr(conf, "wpe_mode", None),
            use_rope=conf.use_rope,
            rope_theta=conf.rope_theta,
        )
    else:
        raise NotImplementedError

    return model

class GeneralTransformerModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=4,
        linear_embedding=False,
        use_wpe=False,
        wpe_mode=None,
        use_rope=False,
        rope_theta=10000.0,
    ):
        super(GeneralTransformerModel, self).__init__()
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
        if wpe_mode is None:
            self.wpe_mode = "all" if use_wpe else "none"
        else:
            self.wpe_mode = wpe_mode
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


    def forward_no_position(self, zs, attention_mask = None):
        embeds = self._read_in(zs)
        output = self._backbone.forward_no_position(inputs_embeds=embeds, attention_mask = attention_mask).last_hidden_state
        prediction = self._read_out(output)
        return prediction
    
    def _should_add_wpe(self, add_wpe=None, step_idx=None):
        use_wpe = getattr(self, "use_wpe", False)
        if not use_wpe:
            return False
        if add_wpe is not None and not add_wpe:
            return False

        wpe_mode = getattr(self, "wpe_mode", "all")
        if wpe_mode == "none":
            return False
        if wpe_mode == "all":
            return True
        if wpe_mode == "once":
            return step_idx in (None, 0)
        raise ValueError(f"Unsupported wpe_mode: {wpe_mode}")

    def forward_single(self, embeds, attention_mask=None, add_wpe=None, step_idx=None):
        if self._should_add_wpe(add_wpe=add_wpe, step_idx=step_idx):
            batch_size, seq_len, _ = embeds.shape
            position_ids = torch.arange(seq_len, device=embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_ids = position_ids % self.n_positions
            output = self._backbone(
                inputs_embeds=embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        else:
            output = self._backbone.forward_no_position(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        return output
    
    def looped_forward(self, zs, horizon, attention_mask = None):
        # input injection
        zs = self._read_in(zs)
        output = torch.zeros_like(zs).to(zs.device)
        output_list = []
        use_wpe = getattr(self, "use_wpe", False)
        for i in range(horizon):
            output = self.forward_single(output+zs, attention_mask, add_wpe=use_wpe, step_idx=i)
            output_list.append(self._read_out(output).clone())
        return output_list

    def looped_forward_without(self, zs, horizon, attention_mask = None):
        # no injection
        output = self._read_in(zs)
        output_list = []
        use_wpe = getattr(self, "use_wpe", False)
        for i in range(horizon):
            output = self.forward_single(output, attention_mask, add_wpe=use_wpe, step_idx=i)
            output_list.append(self._read_out(output).clone())
        return output_list
    
