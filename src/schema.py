from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
    tlist
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "linear_embedding": merge(tboolean, default(False)),
    "use_wpe": merge(tboolean, default(False)),
    "wpe_mode": merge(tstring, nullable, allowed(["none", "once", "all"]), default(None)),
    "use_rope": merge(tboolean, default(False)),
    "rope_theta": merge(tfloat, default(10000.0)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "points": stdict(curriculum_base_schema),
}

training_schema = {
    "task": merge(tstring, allowed(["parity", "sum_reverse", "copy", "dict", "multi", "addition", "mod_add", "mod_add_digits"]), default("parity")),
    "batch_size": merge(tinteger, default(64)),
    "test_len": merge(tinteger, default(10)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "curriculum": stdict(curriculum_schema),
    "ema": merge(tboolean, default(False)),
    "modulus": merge(tinteger, default(11)),
}

wandb_schema = {
    "project": merge(tstring, default("")),
    "entity": merge(tstring, default("")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
