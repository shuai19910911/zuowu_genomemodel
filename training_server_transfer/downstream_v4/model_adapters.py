"""Runtime capability matrix and loaders for public plant DNA models."""

import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_RUNTIME_SPECS = {
    "AgroNT_1B": {"context_bp": 512, "max_tokens": 514, "batch_size": 2, "model_head": "masked-lm-base", "trust_remote_code": False, "zero_shot": False, "token_aligned": False},
    "PlantCaduceus_l32": {"context_bp": 512, "max_tokens": 514, "batch_size": 8, "model_head": "auto", "trust_remote_code": True, "zero_shot": True, "token_aligned": True},
    "PlantCAD2_Small": {"context_bp": 8192, "max_tokens": 8192, "batch_size": 1, "model_head": "auto", "trust_remote_code": True, "zero_shot": True, "token_aligned": True},
    "PlantCAD2_Large": {"context_bp": 8192, "max_tokens": 8192, "batch_size": 1, "model_head": "auto", "trust_remote_code": True, "zero_shot": True, "token_aligned": True},
    "PlantBiMoE": {"context_bp": 8192, "max_tokens": 8194, "batch_size": 1, "model_head": "masked-lm-base", "trust_remote_code": False, "zero_shot": True, "token_aligned": True},
    "GPN_Brassicales": {"context_bp": 512, "max_tokens": 512, "batch_size": 32, "model_head": "auto", "trust_remote_code": False, "zero_shot": True, "token_aligned": True},
    "PlantDNAMamba_BPE": {"context_bp": 1024, "max_tokens": 1024, "batch_size": 4, "model_head": "auto", "trust_remote_code": True, "zero_shot": False, "token_aligned": False},
    "PlantNT_singlebase": {"context_bp": 512, "max_tokens": 514, "batch_size": 8, "model_head": "masked-lm-base", "trust_remote_code": True, "zero_shot": True, "token_aligned": True, "load_dtype": "float32", "inference_dtype": "float32"},
    "NTv2_500M_multi_species": {"context_bp": 512, "max_tokens": 514, "batch_size": 4, "model_head": "masked-lm-base", "trust_remote_code": True, "zero_shot": False, "token_aligned": False, "load_dtype": "float32", "inference_dtype": "float32"},
    "DNABERT2": {"context_bp": 512, "max_tokens": 512, "batch_size": 8, "model_head": "auto", "trust_remote_code": True, "zero_shot": False, "token_aligned": False, "disable_remote_triton": True},
    "GENA_LM_BERT_base": {"context_bp": 512, "max_tokens": 512, "batch_size": 8, "model_head": "auto", "trust_remote_code": True, "zero_shot": False, "token_aligned": False},
    "Caduceus_Ph_131k_d_model256": {"context_bp": 8192, "max_tokens": 8194, "batch_size": 2, "model_head": "auto", "trust_remote_code": True, "zero_shot": True, "token_aligned": True},
    "HyenaDNA_medium_160k": {"context_bp": 8192, "max_tokens": 8194, "batch_size": 2, "model_head": "auto", "trust_remote_code": True, "zero_shot": False, "token_aligned": True, "load_dtype": "float32"},
    "Evo2_1B_base": {"context_bp": 8192, "max_tokens": 8192, "batch_size": 1, "model_head": "evo2", "trust_remote_code": False, "zero_shot": False, "token_aligned": False},
}


@dataclass
class PluginRegistration:
    tokenizer: object = None
    force_trust_remote_code: bool = None
    mode: str = "standard_auto"
    config_overrides: object = None


def clean_model_runtime_cache(model_path):
    """Remove local runtime caches that are not part of the downloaded snapshot."""
    model_path = Path(model_path).resolve()
    cache = model_path / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache)
    for directory in list(model_path.rglob("__pycache__")):
        if directory.is_dir():
            shutil.rmtree(directory)
    for bytecode in list(model_path.rglob("*.pyc")):
        bytecode.unlink(missing_ok=True)


def register_model_plugin(model_path):
    """Register custom configs before Auto* loading; never mutates the model snapshot."""
    model_path = Path(model_path).resolve()
    sys.dont_write_bytecode = True
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    if model_type == "plantbimoe":
        sys.path.insert(0, str(model_path))
        from plantbimoe.configuration_plantbimoe import PlantbimoeConfig
        modeling = importlib.import_module("plantbimoe.modeling_plantbimoe")
        import torch
        class TorchRMSNorm(torch.nn.Module):
            def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
                self.eps = eps
            def forward(self, values):
                variance = values.float().pow(2).mean(dim=-1, keepdim=True)
                normalised = values * torch.rsqrt(variance + self.eps).to(values.dtype)
                return normalised * self.weight
        modeling.RMSNorm = TorchRMSNorm
        Plantbimoe = modeling.Plantbimoe
        PlantbimoeForMaskedLM = modeling.PlantbimoeForMaskedLM
        from plantbimoe.tokenization_plantbimoe import PlantbimoeTokenizer
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
        AutoConfig.register("plantbimoe", PlantbimoeConfig, exist_ok=True)
        AutoModel.register(PlantbimoeConfig, Plantbimoe, exist_ok=True)
        AutoModelForMaskedLM.register(PlantbimoeConfig, PlantbimoeForMaskedLM, exist_ok=True)
        tokenizer = PlantbimoeTokenizer(model_max_length=8192)
        return PluginRegistration(
            tokenizer=tokenizer, force_trust_remote_code=False,
            mode="plantbimoe_local_registration_torch_rmsnorm",
            config_overrides={"fused_add_norm": False},
        )
    if model_type.lower() in {"convnet", "gpn"}:
        importlib.import_module("gpn.model")
        return PluginRegistration(force_trust_remote_code=False, mode="gpn_official_registration")
    return PluginRegistration()


def runtime_spec(model_id):
    if model_id not in MODEL_RUNTIME_SPECS:
        raise ValueError(f"no runtime specification for public model: {model_id}")
    return dict(MODEL_RUNTIME_SPECS[model_id])
