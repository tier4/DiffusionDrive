"""Shared fail-hard checkpoint loading for B2D DiffusionDrive evaluation."""
from typing import Dict, Tuple

import torch

from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper

LIGHTNING_PREFIX = "agent._transfuser_model."


def clean_lightning_state_dict(raw_state_dict: Dict) -> Dict:
    """Strip Lightning prefixes: agent._transfuser_model.xxx -> xxx, agent.xxx -> xxx."""
    cleaned = {}
    for k, v in raw_state_dict.items():
        if k.startswith(LIGHTNING_PREFIX):
            cleaned[k[len(LIGHTNING_PREFIX):]] = v
        elif k.startswith("agent."):
            cleaned[k[len("agent."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_b2d_model(
    checkpoint_path: str,
    config: Bench2DriveConfig,
    device: str,
    allow_partial_load: bool = False,
) -> Tuple[V2TransfuserModelWrapper, Dict]:
    """Build V2TransfuserModelWrapper and load a Lightning checkpoint, failing hard
    on any missing/unexpected key unless allow_partial_load is set."""
    model = V2TransfuserModelWrapper(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = clean_lightning_state_dict(raw_state_dict)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    load_info = {"missing": list(missing), "unexpected": list(unexpected)}
    if missing or unexpected:
        msg = (
            f"Checkpoint {checkpoint_path} did not load cleanly: "
            f"{len(missing)} missing keys {list(missing)[:5]}, "
            f"{len(unexpected)} unexpected keys {list(unexpected)[:5]}."
        )
        if not allow_partial_load:
            raise RuntimeError(msg + " Pass allow_partial_load=True to override.")
        print(f"WARNING: {msg} Continuing due to allow_partial_load.")

    model = model.to(device)
    model.eval()
    return model, load_info
