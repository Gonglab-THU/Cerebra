#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Cerebra inference script.

Examples:
    python inference_model.py \
        --fasta target.fasta \
        --model_id 1 \
        --output output_dir/

    python inference_model.py \
        --a3m target.a3m \
        --checkpoint checkpoint/model_3.pt \
        --output target_model3.pdb \
        --relax

    python inference_model.py \
        --a3m_dir a3ms/ \
        --model_id 1 \
        --output predictions/

    python inference_model.py \
        --fasta_dir fastas/ \
        --model_id 1 \
        --output predictions/

    python inference_model.py \
        --fasta target.fasta \
        --esm2_model /path/to/esm2_t36_3B_UR50D.pt \
        --model_id 1 \
        --output output_dir/
"""

import argparse
import importlib.util
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from cerebra_model.config import model_config
from cerebra_model.get_all_atoms import hu_model_pred_to_atom14_pos, make_atom14_masks
from cerebra_model.humodel.model_with_conf import AlphaFold
from cerebra_model.humodel.structure_module_v2 import NormQuaternion, NormQuaternionMM
from cerebra_model.np import residue_constants as rc
from cerebra_model.np.protein import from_prediction, to_pdb
from cerebra_model.utils.tensor_utils import batched_gather
from data import build_model_features, read_a3m_input, read_fasta_input


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint"
DEFAULT_SEED = 42
DEFAULT_HF_ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
DEFAULT_ESM2_LAYER = 36
DEFAULT_ESM2_CHUNK_SIZE = 1022
DEFAULT_ESM2_CHUNK_OVERLAP = 128
FASTA_EXTENSIONS = {".fa", ".faa", ".fasta", ".fna"}
A3M_EXTENSIONS = {".a3m"}


class TargetInput(NamedTuple):
    path: Path
    name: str
    sequence: str
    msa: torch.Tensor
    input_type: str


class ESM2Context(NamedTuple):
    backend: str
    model: torch.nn.Module
    tokenizer_or_converter: Any


def setup_seed(seed: int) -> None:
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def bf16_autocast(device: torch.device, enabled: bool = False):
    enabled = enabled and device.type == "cuda"
    kwargs = {"enabled": enabled}
    if enabled:
        kwargs["dtype"] = torch.bfloat16

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", **kwargs)

    from torch.cuda.amp import autocast as cuda_autocast

    return cuda_autocast(**kwargs)


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def discover_fasta_files(fasta_dir: Path) -> List[Path]:
    if not fasta_dir.exists():
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")
    if not fasta_dir.is_dir():
        raise NotADirectoryError(f"--fasta_dir must be a directory: {fasta_dir}")

    fasta_files = sorted(
        path
        for path in fasta_dir.iterdir()
        if path.is_file() and path.suffix.lower() in FASTA_EXTENSIONS
    )
    if not fasta_files:
        extensions = ", ".join(sorted(FASTA_EXTENSIONS))
        raise ValueError(f"No FASTA files with extensions [{extensions}] found in {fasta_dir}.")
    return fasta_files


def discover_a3m_files(a3m_dir: Path) -> List[Path]:
    if not a3m_dir.exists():
        raise FileNotFoundError(f"A3M directory not found: {a3m_dir}")
    if not a3m_dir.is_dir():
        raise NotADirectoryError(f"--a3m_dir must be a directory: {a3m_dir}")

    a3m_files = sorted(
        path
        for path in a3m_dir.iterdir()
        if path.is_file() and path.suffix.lower() in A3M_EXTENSIONS
    )
    if not a3m_files:
        extensions = ", ".join(sorted(A3M_EXTENSIONS))
        raise ValueError(f"No A3M files with extensions [{extensions}] found in {a3m_dir}.")
    return a3m_files


def batchify_single(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    batch = {}
    for key, value in features.items():
        if key == "seq":
            batch[key] = [value]
            continue
        if not torch.is_tensor(value):
            batch[key] = [value]
            continue

        value = value.unsqueeze(0)
        if key in {"target_feat", "residue_index", "msa_feat", "msa_depth"}:
            value = value.transpose(1, 0)
        batch[key] = value
    return batch


def infer_huggingface_esm2_name(esm2_model: Optional[str]) -> str:
    if not esm2_model:
        return DEFAULT_HF_ESM2_MODEL
    return esm2_model


def select_esm2_backend(esm2_model: Optional[str]) -> str:
    if not esm2_model:
        return "huggingface"

    model_path = Path(esm2_model)
    if model_path.suffix == ".pt":
        if not model_path.exists():
            raise FileNotFoundError(f"ESM2 .pt model file not found: {model_path}")
        if importlib.util.find_spec("esm") is None:
            raise ImportError(
                "fair-esm package is not installed. Omit --esm2_model to use "
                f"HuggingFace {DEFAULT_HF_ESM2_MODEL}, or install fair-esm."
            )
        return "fair-esm"

    return "huggingface"


def load_fair_esm2_model(
    esm2_model_path: Path,
    device: torch.device,
):
    import esm as fair_esm

    model_data = torch_load_cpu(esm2_model_path)

    regression_data = None
    contact_regression_path = Path(str(esm2_model_path.with_suffix("")) + "-contact-regression.pt")
    if contact_regression_path.exists():
        regression_data = torch_load_cpu(contact_regression_path)

    esm_model, alphabet = fair_esm.pretrained.load_model_and_alphabet_core(
        esm2_model_path.stem,
        model_data,
        regression_data,
    )
    esm_model.eval()
    esm_model.requires_grad_(False)
    esm_model.to(device)
    return ESM2Context("fair-esm", esm_model, alphabet.get_batch_converter())


def load_huggingface_esm2_model(
    hf_model_name_or_path: str,
    device: torch.device,
) -> ESM2Context:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "HuggingFace ESM2 backend requires the transformers package. "
            "Install transformers, or pass a local fair-esm .pt file with --esm2_model."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)
    esm_model = AutoModel.from_pretrained(hf_model_name_or_path)
    esm_model.eval()
    esm_model.requires_grad_(False)
    esm_model.to(device)
    return ESM2Context("huggingface", esm_model, tokenizer)


def load_esm2_model(
    esm2_model: Optional[str],
    device: torch.device,
) -> ESM2Context:
    selected_backend = select_esm2_backend(esm2_model)

    if selected_backend == "fair-esm":
        if esm2_model is None:
            raise ValueError("--esm2_model is required when using fair-esm.")
        esm2_model_path = Path(esm2_model)
        print(f"ESM2 backend: fair-esm ({esm2_model_path})")
        return load_fair_esm2_model(esm2_model_path, device)

    hf_model_name = infer_huggingface_esm2_name(esm2_model)
    print(f"ESM2 backend: HuggingFace transformers ({hf_model_name})")
    return load_huggingface_esm2_model(hf_model_name, device)


def compute_fair_esm2_chunk(
    sequence: str,
    name: str,
    context: ESM2Context,
    device: torch.device,
    repr_layer: int,
) -> torch.Tensor:
    batch_converter = context.tokenizer_or_converter
    with torch.no_grad():
        _, _, batch_tokens = batch_converter([(name, sequence)])
        batch_tokens = batch_tokens.to(device)
        results = context.model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        if repr_layer not in results["representations"]:
            raise KeyError(f"ESM2 representation layer {repr_layer} was not returned.")
        token_embeds = results["representations"][repr_layer]
        token_embeds = token_embeds[:, 1:-1, :].to(dtype=torch.float32).cpu()
    return token_embeds[0, : len(sequence)]


def compute_huggingface_esm2_chunk(
    sequence: str,
    context: ESM2Context,
    device: torch.device,
    repr_layer: int,
) -> torch.Tensor:
    tokenizer = context.tokenizer_or_converter
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        allowed_input_keys = {"input_ids", "attention_mask", "position_ids"}
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in allowed_input_keys
        }
        outputs = context.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("HuggingFace ESM2 did not return hidden states.")
        layer = repr_layer if repr_layer >= 0 else len(hidden_states) + repr_layer
        if layer < 0 or layer >= len(hidden_states):
            raise KeyError(
                f"ESM2 representation layer {repr_layer} is unavailable; "
                f"HuggingFace model returned {len(hidden_states) - 1} transformer layers."
            )
        token_embeds = hidden_states[layer]
        if token_embeds.shape[1] >= len(sequence) + 2:
            token_embeds = token_embeds[:, 1 : 1 + len(sequence), :]
        elif token_embeds.shape[1] >= len(sequence):
            token_embeds = token_embeds[:, : len(sequence), :]
        else:
            raise RuntimeError(
                f"HuggingFace ESM2 returned {token_embeds.shape[1]} tokens for "
                f"{len(sequence)} residues."
            )
        return token_embeds[0].to(dtype=torch.float32).cpu()


def compute_esm2_chunk(
    sequence: str,
    name: str,
    context: ESM2Context,
    device: torch.device,
    repr_layer: int,
) -> torch.Tensor:
    if context.backend == "fair-esm":
        return compute_fair_esm2_chunk(sequence, name, context, device, repr_layer)
    if context.backend == "huggingface":
        return compute_huggingface_esm2_chunk(sequence, context, device, repr_layer)
    raise ValueError(f"Unsupported ESM2 backend: {context.backend}")


def compute_esm2_embedding(
    sequence: str,
    name: str,
    context: ESM2Context,
    device: torch.device,
) -> torch.Tensor:
    repr_layer = DEFAULT_ESM2_LAYER
    chunk_size = DEFAULT_ESM2_CHUNK_SIZE
    chunk_overlap = DEFAULT_ESM2_CHUNK_OVERLAP

    if chunk_size <= 0 or len(sequence) <= chunk_size:
        return compute_esm2_chunk(sequence, name, context, device, repr_layer)
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("ESM2 chunk overlap must be >= 0 and smaller than chunk size.")

    step = chunk_size - chunk_overlap
    starts = list(range(0, len(sequence), step))
    final_start = max(0, len(sequence) - chunk_size)
    starts.append(final_start)
    starts = sorted(set(starts))

    embed_sum = None
    counts = torch.zeros(len(sequence), 1, dtype=torch.float32)
    print(
        f"Sequence length {len(sequence)} exceeds ESM2 chunk size {chunk_size}; "
        f"using {len(starts)} overlapping chunks."
    )
    for chunk_no, start in enumerate(starts, start=1):
        end = min(start + chunk_size, len(sequence))
        chunk_name = f"{name}_{start + 1}_{end}"
        chunk_embedding = compute_esm2_chunk(
            sequence[start:end],
            chunk_name,
            context,
            device,
            repr_layer,
        )
        if embed_sum is None:
            embed_sum = torch.zeros(len(sequence), chunk_embedding.shape[-1], dtype=torch.float32)
        embed_sum[start:end] += chunk_embedding
        counts[start:end] += 1.0
        print(f"  ESM2 chunk {chunk_no}/{len(starts)}: residues {start + 1}-{end}")
        clear_cuda_cache(device)

    if torch.any(counts == 0):
        raise RuntimeError("Internal error: some residues did not receive ESM2 embeddings.")
    return embed_sum / counts


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("ema_state", "model_state", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        ema_state = checkpoint.get("ema")
        if isinstance(ema_state, dict) and isinstance(ema_state.get("params"), dict):
            return ema_state["params"]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find a PyTorch state_dict in the checkpoint.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    module_prefix_count = sum(key.startswith("module.") for key in keys)
    if module_prefix_count <= len(keys) // 2:
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def load_cerebra_model(
    checkpoint_path: Path,
    device: torch.device,
    config_preset: str,
    low_prec: bool,
) -> AlphaFold:
    config = model_config(name=config_preset, train=False, low_prec=low_prec)
    model = AlphaFold(config)
    checkpoint = torch_load_cpu(checkpoint_path)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))

    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_keys = {"esm2_boost_logit"}
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    unexpected_keys_to_ignore = {
        key
        for key in unexpected_keys
        if key.startswith("conf_head.") or key.startswith("structure.pLDDT.")
    }

    missing_keys_to_report = missing_keys - allowed_missing_keys
    unexpected_keys_to_report = unexpected_keys - unexpected_keys_to_ignore
    if missing_keys_to_report or unexpected_keys_to_report:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility. "
            f"Missing: {sorted(missing_keys_to_report)}; "
            f"unexpected: {sorted(unexpected_keys_to_report)}"
        )

    if "esm2_boost_logit" in missing_keys:
        with torch.no_grad():
            model.esm2_boost_logit.fill_(float("-inf"))
        print("Checkpoint missing esm2_boost_logit; setting esm2_boost to 1.0.")
    if unexpected_keys_to_ignore:
        print(f"Ignoring {len(unexpected_keys_to_ignore)} checkpoint keys for disabled heads.")

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Loaded Cerebra checkpoint: {checkpoint_path}")
    print(f"{total_params:,} total parameters.")
    return model


def make_anchor_list(length: int) -> np.ndarray:
    if length < 8:
        raise ValueError(f"Target length must be at least 8 residues, got {length}.")

    n_clusters = 24
    if 324 >= length >= 224:
        n_clusters = 32
    if length > 324:
        n_clusters = 48
    if length > 500:
        n_clusters = 56

    anchors = np.array([int(x * (length - 8) / n_clusters) for x in range(n_clusters)]) + 5
    anchors = anchors.astype(int)
    return np.clip(anchors, a_min=2, a_max=length - 2)


def run_cycle(
    batch: Dict[str, torch.Tensor],
    model: AlphaFold,
    use_bf16: bool,
    return_dist: bool,
    conf_anchor_chunk_size: int,
):
    prevs = [None, None, None]

    dims = batch["msa_feat"].shape
    num_iters = dims[0]
    batch_size = dims[1]
    msa_rows = dims[2]
    length = dims[3]
    anchor_list = make_anchor_list(length)

    with torch.no_grad():
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        for cycle_no in range(num_iters):
            is_final_iter = cycle_no == (num_iters - 1)
            feats = {
                "seq_mask": torch.ones([batch_size, length], dtype=torch.float32),
                "msa_mask": torch.ones([batch_size, msa_rows, length], dtype=torch.float32),
                "target_feat": batch["target_feat"][cycle_no],
                "residue_index": batch["residue_index"][cycle_no],
                "msa_feat": batch["msa_feat"][cycle_no],
                "esm2": batch["esm2"],
            }
            if "msa_depth" in batch:
                feats["msa_depth"] = batch["msa_depth"][cycle_no]

            for key, value in list(feats.items()):
                if value.dtype == torch.float32:
                    value = value.to(dtype=dtype)
                feats[key] = value.to(device)

            with bf16_autocast(device, enabled=use_bf16):
                m_1_prev, z_prev, x_prev, outputs = model(
                    feats,
                    prevs,
                    anchor_list,
                    _recycle=(num_iters > 1),
                    return_aux=False,
                    return_dist=(return_dist and is_final_iter),
                    return_conf=is_final_iter,
                    return_angles=is_final_iter,
                    reduce_plddt=True,
                    conf_anchor_chunk_size=conf_anchor_chunk_size,
                    keep_structure_all=False,
                )

            if is_final_iter:
                return outputs

            prevs = [m_1_prev.detach(), z_prev.detach(), x_prev.detach()]
            del outputs, feats
            clear_cuda_cache(device)

    raise RuntimeError("Model did not return outputs.")


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width,
        end=1.0,
        step=bin_width,
        device=logits.device,
        dtype=logits.dtype,
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), num_bins),
        dim=-1,
    )


def reduce_plddt(raw_plddt: torch.Tensor) -> torch.Tensor:
    if raw_plddt.dim() == 2:
        return raw_plddt
    if raw_plddt.dim() == 3:
        return raw_plddt.mean(dim=1)
    if raw_plddt.shape[-1] == 50:
        plddt = compute_plddt(raw_plddt)
        while plddt.dim() > 2:
            plddt = plddt.mean(dim=1)
        return plddt
    raise ValueError(f"Unsupported pLDDT tensor shape: {tuple(raw_plddt.shape)}")


def rotation_to_quaternion(rotation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r00, r11, r22 = rotation[..., 0, 0], rotation[..., 1, 1], rotation[..., 2, 2]
    trace = r00 + r11 + r22
    safe_mask = (trace + 1.0) > eps
    quaternion = torch.zeros(*rotation.shape[:-2], 4, device=rotation.device, dtype=rotation.dtype)

    if safe_mask.any():
        sub_r = rotation[safe_mask]
        sub_trace = trace[safe_mask]
        a = torch.sqrt(sub_trace + 1.0) * 0.5
        denom = 4.0 * a
        b = (sub_r[:, 2, 1] - sub_r[:, 1, 2]) / denom
        c = (sub_r[:, 0, 2] - sub_r[:, 2, 0]) / denom
        d = (sub_r[:, 1, 0] - sub_r[:, 0, 1]) / denom
        quaternion[safe_mask] = torch.stack([a, b, c, d], dim=-1)

    unsafe_mask = ~safe_mask
    if unsafe_mask.any():
        sub_r = rotation[unsafe_mask]
        sub_r00 = sub_r[:, 0, 0]
        sub_r11 = sub_r[:, 1, 1]
        sub_r22 = sub_r[:, 2, 2]
        candidates = torch.stack(
            [
                1.0 + sub_r00 - sub_r11 - sub_r22,
                1.0 - sub_r00 + sub_r11 - sub_r22,
                1.0 - sub_r00 - sub_r11 + sub_r22,
            ],
            dim=-1,
        )
        vals, idx = torch.max(candidates, dim=-1)
        t = torch.sqrt(torch.relu(vals))
        t_inv = 0.5 / (t + 1e-8)
        t = 0.5 * t
        sub_q = torch.zeros(sub_r.shape[0], 4, device=rotation.device, dtype=rotation.dtype)

        mask_x = idx == 0
        if mask_x.any():
            sub_q[mask_x, 0] = (sub_r[mask_x, 2, 1] - sub_r[mask_x, 1, 2]) * t_inv[mask_x]
            sub_q[mask_x, 1] = t[mask_x]
            sub_q[mask_x, 2] = (sub_r[mask_x, 1, 0] + sub_r[mask_x, 0, 1]) * t_inv[mask_x]
            sub_q[mask_x, 3] = (sub_r[mask_x, 0, 2] + sub_r[mask_x, 2, 0]) * t_inv[mask_x]

        mask_y = idx == 1
        if mask_y.any():
            sub_q[mask_y, 0] = (sub_r[mask_y, 0, 2] - sub_r[mask_y, 2, 0]) * t_inv[mask_y]
            sub_q[mask_y, 1] = (sub_r[mask_y, 1, 0] + sub_r[mask_y, 0, 1]) * t_inv[mask_y]
            sub_q[mask_y, 2] = t[mask_y]
            sub_q[mask_y, 3] = (sub_r[mask_y, 2, 1] + sub_r[mask_y, 1, 2]) * t_inv[mask_y]

        mask_z = idx == 2
        if mask_z.any():
            sub_q[mask_z, 0] = (sub_r[mask_z, 1, 0] - sub_r[mask_z, 0, 1]) * t_inv[mask_z]
            sub_q[mask_z, 1] = (sub_r[mask_z, 2, 0] + sub_r[mask_z, 0, 2]) * t_inv[mask_z]
            sub_q[mask_z, 2] = (sub_r[mask_z, 2, 1] + sub_r[mask_z, 1, 2]) * t_inv[mask_z]
            sub_q[mask_z, 3] = t[mask_z]

        quaternion[unsafe_mask] = sub_q

    return NormQuaternion(quaternion)


def compute_consensus_positions(positions: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    top_k = min(top_k, positions.shape[0])
    mean_pos = positions.mean(dim=0)
    distances = torch.abs(positions - mean_pos).mean(dim=-1)
    topk_val, _ = distances.topk(k=top_k, largest=False)
    threshold = topk_val[-1]
    return positions[distances <= threshold].mean(dim=0)


def get_transformation(mobile: np.ndarray, target: np.ndarray, return_rotation: bool = False):
    mobile_center = mobile.mean(0)
    target_center = target.mean(0)
    mobile = mobile - mobile_center
    target = target - target_center
    matrix = np.dot(mobile.T, target)

    u, _, vh = np.linalg.svd(matrix)
    determinant_sign = np.sign(np.linalg.det(np.dot(u, vh)))
    identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, determinant_sign]])
    rotation = np.dot(vh.T, np.dot(identity, u.T))
    transformed = np.dot(target, rotation)
    if return_rotation:
        return transformed, rotation
    return transformed


def prediction_to_atom14(pred_output, batch, use_consensus: bool):
    device = pred_output["translation"][-1].device
    if use_consensus:
        xyz = pred_output["translation"][-1][0].detach().cpu().numpy()
        quaternion = pred_output["quaternion"][-1]
        anchor_id = xyz.shape[0] // 2
        main_anchor_pos = xyz[anchor_id]
        translations = []
        rotations = []

        for anchor_idx in range(xyz.shape[0]):
            transformed, rotation = get_transformation(main_anchor_pos, xyz[anchor_idx], return_rotation=True)
            translations.append(transformed)
            rotations.append(
                rotation_to_quaternion(
                    torch.as_tensor(rotation, device=device, dtype=quaternion.dtype)
                )
            )

        translations = torch.as_tensor(np.array(translations), device=device).permute(1, 0, 2)
        translations = translations.to(dtype=quaternion.dtype)
        consensus_translations = torch.stack(
            [compute_consensus_positions(translated) for translated in translations]
        )

        rotations = torch.stack(rotations)[None, :, None, :]
        rotations = rotations.to(device=quaternion.device, dtype=quaternion.dtype)
        combined_quaternion = NormQuaternionMM(rotations, quaternion)
        combined_quaternion = combined_quaternion[0].permute(1, 0, 2)
        consensus_rotations = torch.stack(
            [compute_consensus_positions(rotation) for rotation in combined_quaternion]
        )

        quaternion = NormQuaternion(consensus_rotations.unsqueeze(0))
        translation = consensus_translations.unsqueeze(0)
    else:
        anchor_id = pred_output["translation"][-1].shape[1] // 2
        quaternion = pred_output["quaternion"][-1][:, anchor_id]
        translation = pred_output["translation"][-1][:, anchor_id]

    angles = pred_output["angles"][-1].to(device)
    plddt = reduce_plddt(pred_output["pLDDT"])

    raw_seq = batch["true_msa"][0, 0]
    openfold_seq = [
        rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[int(aa.item())]
        for aa in raw_seq
    ]
    openfold_seq = torch.LongTensor(openfold_seq).unsqueeze(0).to(device)

    all_atom_pos_14 = hu_model_pred_to_atom14_pos(
        quaternion,
        translation,
        angles,
        openfold_seq,
    )
    batch["aatype"] = openfold_seq
    batch = make_atom14_masks(batch)
    return all_atom_pos_14, plddt, batch


def write_pdb(
    pred_all_atoms: torch.Tensor,
    plddt: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    output_path: Path,
    relax: bool,
    relax_device: str,
):
    num_batch, length = pred_all_atoms.shape[:2]
    pred_all_atoms = batched_gather(
        pred_all_atoms,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(pred_all_atoms.shape[:-2]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    relaxed_path = None
    for batch_idx in range(num_batch):
        pdb_write = {
            "residue_index": np.arange(length),
            "aatype": batch["aatype"][batch_idx].detach().cpu().numpy(),
            "final_atom_positions": pred_all_atoms[batch_idx].detach().cpu().numpy(),
            "final_atom_mask": batch["atom37_atom_exists"][batch_idx].detach().cpu().numpy(),
        }
        b_factors = plddt[batch_idx].unsqueeze(-1).repeat(1, 37) * 100
        b_factors = torch.clip(b_factors, min=0, max=99.99).detach().cpu().numpy()
        protein = from_prediction(pdb_write, pdb_write, b_factors)

        with output_path.open("w") as handle:
            handle.write(to_pdb(protein))

        if relax:
            relax_config = model_config(name="model_3", train=False, low_prec=True)
            relax_output_name = output_path.stem
            run_relax(relax_config, relax_device, protein, output_path.parent, relax_output_name)
            relaxed_path = output_path.parent / f"{relax_output_name}_relaxed.pdb"

    return relaxed_path


def run_relax(relax_config, relax_device: str, protein, output_dir: Path, output_name: str) -> None:
    from cerebra_model.np.relax import relax as relax_module

    def relax_once(device_name: str) -> str:
        device_name = (device_name or "cpu").lower()
        use_gpu = device_name != "cpu"
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        try:
            if device_name.startswith("cuda:"):
                os.environ["CUDA_VISIBLE_DEVICES"] = device_name.split(":")[-1]

            amber_relaxer = relax_module.AmberRelaxation(
                use_gpu=use_gpu,
                **relax_config.relax,
            )
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=protein)
            return relaxed_pdb_str
        finally:
            if visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    try:
        relaxed_pdb_str = relax_once(relax_device)
    except Exception as exc:
        if (relax_device or "cpu").lower() == "cpu":
            raise
        print(f"Relax failed on {relax_device}: {exc}")
        print("Retrying relax on CPU.")
        relaxed_pdb_str = relax_once("cpu")

    relaxed_output_path = output_dir / f"{output_name}_relaxed.pdb"
    with relaxed_output_path.open("w") as handle:
        handle.write(relaxed_pdb_str)


def resolve_checkpoint_path(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    return args.checkpoint_dir / f"model_{args.model_id}.pt"


def resolve_output_path(
    output: Path,
    target_name: str,
    model_id: int,
    force_directory: bool = False,
) -> Path:
    if force_directory:
        if output.exists() and not output.is_dir():
            raise ValueError("--output must be a directory when --fasta_dir or --a3m_dir is used.")
        return output / f"{target_name}_model_{model_id}.pdb"

    output_str = str(output)
    if output_str.endswith(os.sep) or (output.exists() and output.is_dir()) or output.suffix == "":
        return output / f"{target_name}_model_{model_id}.pdb"
    return output


def resolve_output_paths(
    output: Path,
    targets: Sequence[TargetInput],
    model_id: int,
    force_directory: bool,
) -> List[Path]:
    output_paths: List[Path] = []
    used_paths: Dict[str, int] = {}

    for target in targets:
        name = target.name
        output_path = resolve_output_path(output, name, model_id, force_directory=force_directory)
        output_key = str(output_path.resolve())
        suffix_no = 1
        while output_key in used_paths:
            suffix_no += 1
            output_path = resolve_output_path(
                output,
                f"{target.name}_{suffix_no}",
                model_id,
                force_directory=force_directory,
            )
            output_key = str(output_path.resolve())
        used_paths[output_key] = 1
        output_paths.append(output_path.resolve())

    return output_paths


def resolve_device(args) -> torch.device:
    if args.cuda_device is not None:
        device = torch.device(f"cuda:{args.cuda_device}")
    elif args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device)
    return device


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision == "fp32":
        return "fp32"

    if precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("--precision bf16 requires a CUDA device. Use --precision auto or fp32 on CPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Current GPU does not support bf16 autocast. Use --precision auto or fp32."
            )
        return "bf16"

    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"

    raise ValueError(f"Unsupported precision: {precision}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run Cerebra inference from FASTA or A3M inputs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fasta", type=Path, help="Input FASTA file for single-sequence inference.")
    input_group.add_argument(
        "--fasta_dir",
        type=Path,
        help="Directory of FASTA files for batch single-sequence inference.",
    )
    input_group.add_argument("--a3m", type=Path, help="Input A3M file for MSA-based inference.")
    input_group.add_argument(
        "--a3m_dir",
        type=Path,
        help="Directory of A3M files for batch MSA-based inference.",
    )

    parser.add_argument(
        "--esm2_model",
        "--esm2-param",
        dest="esm2_model",
        type=str,
        default=None,
        help=(
            f"Optional ESM2 source. Omit to use HuggingFace {DEFAULT_HF_ESM2_MODEL}; "
            "pass a local .pt file for fair-esm, or a HuggingFace model id/local directory."
        ),
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing model_1.pt, model_2.pt, ... checkpoints.",
    )
    parser.add_argument("--model_id", "--model", dest="model_id", type=int, default=1, help="Checkpoint id to load.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit Cerebra checkpoint path. Overrides --checkpoint_dir and --model_id.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output PDB file or output directory.")

    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--cuda_device", type=int, default=None, help="CUDA device id. Overrides --device.")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "auto"),
        default="auto",
        help="Cerebra inference precision. auto uses bf16 on supported CUDA GPUs, otherwise fp32.",
    )
    parser.add_argument("--cycles", type=int, default=4, help="Number of recycle cycles.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--torch_threads", type=int, default=20, help="torch.set_num_threads value.")
    parser.add_argument("--max_msa_sequences", type=int, default=2000, help="Maximum A3M rows to keep; <=0 keeps all rows.")
    parser.add_argument("--sample_msa_sequences", type=int, default=256, help="MSA rows sampled before clustering.")
    parser.add_argument("--min_msa_rows", type=int, default=128, help="Pad shallow MSAs to at least this many rows.")
    parser.add_argument("--conf_anchor_chunk_size", type=int, default=32, help="Chunk size for confidence-head anchors.")
    parser.add_argument("--config_preset", type=str, default="model_3", help="Cerebra model config preset.")
    parser.add_argument("--relax", action="store_true", help="Run Amber relax after writing the unrelaxed PDB.")
    parser.add_argument(
        "--relax_device",
        type=str,
        default=None,
        help="Relax device. Defaults to cpu. Use cuda:0 only when OpenMM CUDA is compatible.",
    )
    parser.add_argument(
        "--no_consensus",
        action="store_true",
        help="Use the middle anchor directly instead of anchor consensus.",
    )
    parser.add_argument(
        "--return_dist",
        action="store_true",
        help="Also compute the distance head during the final recycle. Not written by default.",
    )
    return parser.parse_args(argv)


def run_model_with_precision(
    batch: Dict[str, torch.Tensor],
    model: AlphaFold,
    precision: str,
    return_dist: bool,
    conf_anchor_chunk_size: int,
):
    use_bf16 = precision == "bf16"
    return run_cycle(
        batch,
        model,
        use_bf16=use_bf16,
        return_dist=return_dist,
        conf_anchor_chunk_size=conf_anchor_chunk_size,
    )


def load_inputs(args) -> List[TargetInput]:
    if args.fasta is not None:
        if not args.fasta.exists():
            raise FileNotFoundError(f"Input FASTA file not found: {args.fasta}")
        name, sequence, msa = read_fasta_input(args.fasta)
        return [TargetInput(args.fasta, name, sequence, msa, "fasta")]

    if args.fasta_dir is not None:
        targets = []
        for fasta_path in discover_fasta_files(args.fasta_dir):
            name, sequence, msa = read_fasta_input(fasta_path)
            targets.append(TargetInput(fasta_path, name, sequence, msa, "fasta"))
        return targets

    if args.a3m is not None:
        if not args.a3m.exists():
            raise FileNotFoundError(f"Input A3M file not found: {args.a3m}")
        name, sequence, msa = read_a3m_input(args.a3m)
        return [TargetInput(args.a3m, name, sequence, msa, "a3m")]

    if args.a3m_dir is not None:
        targets = []
        for a3m_path in discover_a3m_files(args.a3m_dir):
            name, sequence, msa = read_a3m_input(a3m_path)
            targets.append(TargetInput(a3m_path, name, sequence, msa, "a3m"))
        return targets

    raise ValueError("One of --fasta, --fasta_dir, --a3m, or --a3m_dir is required.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    setup_seed(args.seed)

    device = resolve_device(args)
    effective_precision = resolve_precision(args.precision, device)

    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cerebra checkpoint not found: {checkpoint_path}")

    targets = load_inputs(args)
    output_paths = resolve_output_paths(
        args.output,
        targets,
        args.model_id,
        force_directory=(args.fasta_dir is not None or args.a3m_dir is not None),
    )
    relax_device = args.relax_device or "cpu"

    if len(targets) == 1:
        target = targets[0]
        print(f"Input type: {target.input_type}")
        print(f"Target: {target.name}")
        print(f"Length: {len(target.sequence)} residues")
        print(f"MSA rows before filtering: {target.msa.shape[0]}")
        print(f"Output: {output_paths[0]}")
    else:
        input_type = "fasta_dir" if args.fasta_dir is not None else "a3m_dir"
        input_label = "FASTA" if args.fasta_dir is not None else "A3M"
        print(f"Input type: {input_type}")
        print(f"Targets: {len(targets)} {input_label} files")
        print(f"Output directory: {args.output.resolve()}")

    start_time = time.perf_counter()
    print("Loading ESM2...")
    esm2_context = load_esm2_model(
        args.esm2_model,
        device,
    )

    embedded_targets: List[Tuple[TargetInput, Path, torch.Tensor]] = []
    for target_no, (target, output_path) in enumerate(zip(targets, output_paths), start=1):
        if len(targets) > 1:
            print(
                f"[{target_no}/{len(targets)}] ESM2 embedding: {target.name} "
                f"({len(target.sequence)} residues, {target.msa.shape[0]} MSA rows)"
            )
        esm2_embedding = compute_esm2_embedding(
            target.sequence,
            target.name,
            esm2_context,
            device,
        )
        embedded_targets.append((target, output_path, esm2_embedding))

    esm2_context.model.to("cpu")
    del esm2_context
    clear_cuda_cache(device)

    print("Loading Cerebra model...")
    model = load_cerebra_model(
        checkpoint_path,
        device=device,
        config_preset=args.config_preset,
        low_prec=True,
    )

    results = []
    for target_no, (target, output_path, esm2_embedding) in enumerate(embedded_targets, start=1):
        target_start = time.perf_counter()
        if len(targets) > 1:
            print(f"[{target_no}/{len(targets)}] Running Cerebra: {target.name}")
            print(f"Output: {output_path}")

        print("Building model features...")
        features = build_model_features(
            target.msa,
            target.sequence,
            esm2_embedding,
            cycles=args.cycles,
            seed=args.seed,
            max_msa_sequences=args.max_msa_sequences,
            sample_msa_sequences=args.sample_msa_sequences,
            min_msa_rows=args.min_msa_rows,
        )
        batch = batchify_single(features)

        if args.precision == effective_precision:
            print(f"Running Cerebra inference with precision={effective_precision}...")
        else:
            print(f"Running Cerebra inference with precision={args.precision}->{effective_precision}...")
        outputs = run_model_with_precision(
            batch,
            model,
            precision=effective_precision,
            return_dist=args.return_dist,
            conf_anchor_chunk_size=args.conf_anchor_chunk_size,
        )
        all_atom_pos_14, plddt, batch = prediction_to_atom14(
            outputs,
            batch,
            use_consensus=not args.no_consensus,
        )
        relaxed_path = write_pdb(
            all_atom_pos_14,
            plddt,
            batch,
            output_path,
            relax=args.relax,
            relax_device=relax_device,
        )

        target_elapsed = time.perf_counter() - target_start
        mean_plddt = float(plddt.mean() * 100)
        print(f"Unrelaxed PDB written to: {output_path}")
        if relaxed_path is not None:
            print(f"Relaxed PDB written to: {relaxed_path}")
        print(f"Mean pLDDT: {mean_plddt:.2f}")
        if len(targets) > 1:
            print(f"Target done in {target_elapsed:.2f} seconds.")
        results.append((target.name, output_path, relaxed_path, mean_plddt, target_elapsed))

        del features, batch, outputs, all_atom_pos_14, plddt, esm2_embedding
        clear_cuda_cache(device)

    elapsed = time.perf_counter() - start_time
    if len(results) > 1:
        print("Batch summary:")
        for name, output_path, relaxed_path, mean_plddt, _ in results:
            print(f"  {name}: mean pLDDT={mean_plddt:.2f}, pdb={output_path}")
            if relaxed_path is not None:
                print(f"  {name}: relaxed pdb={relaxed_path}")
        print(f"Processed {len(results)} targets in {elapsed:.2f} seconds.")
    else:
        print(f"Done in {elapsed:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
