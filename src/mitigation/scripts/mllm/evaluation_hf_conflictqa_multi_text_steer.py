#!/usr/bin/env python3
"""HF ConflictQA multi-evidence evaluator with text-direction activation steering.

This adapts the Modality-Preference idea to the ConflictQA setting:

1. Estimate a text-vs-image steering direction from either paired instructions
   or baseline behavior tags.
2. Run the original ConflictQA multi-evidence prompt without an explicit
   instruction, while adding that direction to selected decoder layers.

Default task matches ConflictQA code/mcq/multi_text=true_para.py:
text evidence is parametric_memory and image evidence is the counter image.
In that setup, reducing External matches means the model is less often following
that misleading image.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

torch = None
Image = None
tqdm = None
AutoModelForImageTextToText = None
AutoProcessor = None


def import_runtime_deps() -> None:
    global torch, Image, tqdm, AutoModelForImageTextToText, AutoProcessor
    if torch is not None:
        return
    import torch as torch_module
    from PIL import Image as image_module
    from tqdm import tqdm as tqdm_function
    from transformers import AutoModelForImageTextToText as auto_model_class
    from transformers import AutoProcessor as auto_processor_class

    torch = torch_module
    Image = image_module
    tqdm = tqdm_function
    AutoModelForImageTextToText = auto_model_class
    AutoProcessor = auto_processor_class


TEXT_STEER_INSTRUCTION = (
    "In case there is an inconsistency between the Text Context and the input "
    "image content, you must rely on the text context rather than the input "
    "image content."
)
IMAGE_STEER_INSTRUCTION = (
    "In case there is an inconsistency between the text context and the image "
    "content, you should follow the image content rather than the text context."
)
ANSWER_INSTRUCTION = "Answer in 1-5 words. Do not use a full sentence."

TASK_PRESETS = {
    "multi_text_true_para": {
        "text_field": "parametric_memory",
        "image_kind": "counter",
        "output_file": "multi_text=true_para_text_steered_hf.json",
    },
    "multi_internal_true_para": {
        "text_field": None,
        "image_kind": None,
        "image_root": "./data/MMKC/ConflictQA/image",
        "data_path": "./data/MMKC/ConflictQA/data/qwen7b/mcq_internal=true_para_v2.json",
        "orders": "itq1,tiq1,itq2,tiq2",
        "calibration_orders": "itq1,tiq1,itq2,tiq2",
        "output_file": "multi_internal=true_para_text_steered_hf.json",
    },
    "multi_image_true_para": {
        "text_field": "counter_memory",
        "image_kind": "para",
        "output_file": "multi_image=true_para_text_steered_hf.json",
    },
    "multi_text_true_counter": {
        "text_field": "parametric_memory",
        "image_kind": "para",
        "output_file": "multi_text=true_counter_text_steered_hf.json",
    },
    "multi_image_true_counter": {
        "text_field": "counter_memory",
        "image_kind": "counter",
        "output_file": "multi_image=true_counter_text_steered_hf.json",
    },
}

MULTI_INTERNAL_TRUE_MODALITIES = {"itq1", "tiq1", "itq2", "tiq2"}


def parser_default_image_root() -> str:
    return "./data/MMKC/ConflictQA/image/gpt4"


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_order_scales(value: str | None) -> dict[str, float]:
    scales: dict[str, float] = {}
    if not value:
        return scales
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        order, sep, raw_scale = part.partition("=")
        if not sep:
            raise ValueError("--order_steer_scales entries must look like itq=-5,tiq=5")
        order = order.strip()
        if order not in {"itq", "tiq", *MULTI_INTERNAL_TRUE_MODALITIES}:
            raise ValueError(f"Unknown order in --order_steer_scales: {order}")
        scales[order] = float(raw_scale)
    return scales


def base_order(order: str) -> str:
    if order.startswith("itq"):
        return "itq"
    if order.startswith("tiq"):
        return "tiq"
    return order


def scale_for_order(order: str, args: argparse.Namespace) -> float:
    return args.order_steer_scale_map.get(order, args.order_steer_scale_map.get(base_order(order), args.steer_scale))


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")
    return data


def normalize_slice(start: int | None, end: int | None, total: int, label: str) -> tuple[int, int]:
    start_value = 0 if start is None else start
    end_value = total if end is None else end
    if start_value < 0:
        raise ValueError(f"--{label}-start must be non-negative.")
    if end_value < 0:
        raise ValueError(f"--{label}-end must be non-negative.")
    if start_value > total:
        raise ValueError(f"--{label}-start({start_value}) exceeds data size({total}).")
    if end_value > total:
        raise ValueError(f"--{label}-end({end_value}) exceeds data size({total}).")
    if end_value < start_value:
        raise ValueError(f"--{label}-end({end_value}) must be >= --{label}-start({start_value}).")
    return start_value, end_value


def slice_rows_with_indices(
    rows: list[dict[str, Any]],
    *,
    start: int | None,
    end: int | None,
    label: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    start_value, end_value = normalize_slice(start, end, len(rows), label)
    return list(range(start_value, end_value)), rows[start_value:end_value]


def model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    return next(model.parameters()).device


def resolve_attr(obj: Any, dotted_path: str) -> Any:
    current = obj
    for name in dotted_path.split("."):
        current = getattr(current, name)
    return current


def get_decoder_layers(model) -> Any:
    candidate_paths = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.model.model.layers",
    ]
    for path in candidate_paths:
        try:
            layers = resolve_attr(model, path)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise AttributeError("Could not find decoder layers. Tried: " + ", ".join(candidate_paths))


def image_path_for(row: dict[str, Any], image_root: Path, image_kind: str) -> Path:
    if image_kind == "row":
        image_path = row.get("image")
        if not image_path:
            raise KeyError("--image-kind row requires an `image` field in each row.")
        return Path(image_path)
    return image_root / image_kind / f"{image_kind}_{row['original_idx']}.jpg"


def multi_internal_true_image_path(row: dict[str, Any], image_root: Path, modality: str) -> Path:
    if modality in {"itq1", "tiq1"}:
        original_idx2 = row.get("original_idx2", row["original_idx"])
        return image_root / "chat_gpt4" / "counter" / f"counter_{original_idx2}.jpg"
    if modality in {"itq2", "tiq2"}:
        return image_root / "gpt4" / "counter" / f"counter_{row['original_idx']}.jpg"
    raise ValueError(f"Unknown multi_internal_true modality: {modality}")


def prompt_from_memory_question(memory: Any, question: Any, instruction: str | None = None) -> str:
    parts = [str(memory).strip()]
    if instruction:
        parts.append(instruction)
    parts.append(f"Question: {str(question).strip()}")
    parts.append(ANSWER_INSTRUCTION)
    return "\n\n".join(parts)


def prompt_for(row: dict[str, Any], text_field: str, instruction: str | None = None) -> str:
    return prompt_from_memory_question(row[text_field], row["question"], instruction)


def multi_internal_true_prompt(row: dict[str, Any], modality: str, instruction: str | None = None) -> str:
    if modality in {"itq1", "tiq1"}:
        memory = row["counter_memory"]
    elif modality in {"itq2", "tiq2"}:
        memory = row.get("counter_memory2", row.get("counter_memory_aligned_evidence", row["counter_memory"]))
    else:
        raise ValueError(f"Unknown multi_internal_true modality: {modality}")
    return prompt_from_memory_question(memory, row["question"], instruction)


def multimodal_messages(prompt: str, order: str) -> list[dict[str, Any]]:
    order = base_order(order)
    if order == "itq":
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    elif order == "tiq":
        before, sep, after = prompt.partition("\n\nQuestion:")
        if sep:
            content = [
                {"type": "text", "text": before},
                {"type": "image"},
                {"type": "text", "text": "Question:" + after},
            ]
        else:
            content = [{"type": "text", "text": prompt}, {"type": "image"}]
    else:
        raise ValueError(f"Unknown order: {order}")
    return [{"role": "user", "content": content}]


def encode_inputs(processor, model, prompt: str, image: Image.Image, order: str):
    messages = multimodal_messages(prompt, order)
    chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt", padding=True)
    return inputs.to(model_input_device(model))


def last_prompt_hidden(
    processor,
    model,
    row: dict[str, Any],
    *,
    text_field: str,
    image_root: Path,
    image_kind: str,
    order: str,
    instruction: str,
) -> tuple[torch.Tensor, ...]:
    image = Image.open(image_path_for(row, image_root, image_kind)).convert("RGB")
    prompt = prompt_for(row, text_field, instruction)
    inputs = encode_inputs(processor, model, prompt, image, order)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return tuple(hidden[:, -1, :].detach().cpu() for hidden in outputs.hidden_states)


def last_prompt_hidden_multi_internal_true(
    processor,
    model,
    row: dict[str, Any],
    *,
    image_root: Path,
    modality: str,
    instruction: str | None,
) -> tuple[torch.Tensor, ...]:
    image = Image.open(multi_internal_true_image_path(row, image_root, modality)).convert("RGB")
    prompt = multi_internal_true_prompt(row, modality, instruction)
    inputs = encode_inputs(processor, model, prompt, image, modality)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return tuple(hidden[:, -1, :].detach().cpu() for hidden in outputs.hidden_states)


def normalize_vector(vector: torch.Tensor, method: str) -> torch.Tensor:
    if method == "none":
        return vector
    if method == "l2":
        return vector / vector.norm(p=2).clamp_min(1e-6)
    if method == "rms":
        return vector / vector.pow(2).mean().sqrt().clamp_min(1e-6)
    raise ValueError(f"Unknown normalize method: {method}")


def build_steering_vectors(
    processor,
    model,
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    image_root: Path,
    image_kind: str,
    orders: list[str],
    layers: list[int],
    calibration_samples: int,
    normalize: str,
) -> dict[int, torch.Tensor]:
    selected_rows = rows if calibration_samples <= 0 else rows[:calibration_samples]
    if not selected_rows:
        raise ValueError("Need at least one calibration row.")

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {layer: 0 for layer in layers}
    for row in tqdm(selected_rows, desc="building text steering vectors"):
        for order in orders:
            text_hidden = last_prompt_hidden(
                processor,
                model,
                row,
                text_field=text_field,
                image_root=image_root,
                image_kind=image_kind,
                order=order,
                instruction=TEXT_STEER_INSTRUCTION,
            )
            image_hidden = last_prompt_hidden(
                processor,
                model,
                row,
                text_field=text_field,
                image_root=image_root,
                image_kind=image_kind,
                order=order,
                instruction=IMAGE_STEER_INSTRUCTION,
            )
            for layer in layers:
                hidden_index = layer + 1
                if hidden_index >= len(text_hidden):
                    raise IndexError(f"Layer {layer} is out of range for {len(text_hidden) - 1} decoder layers.")
                delta = (text_hidden[hidden_index] - image_hidden[hidden_index]).mean(dim=0)
                sums[layer] = delta if layer not in sums else sums[layer] + delta
                counts[layer] += 1

    return {layer: normalize_vector(sums[layer] / max(counts[layer], 1), normalize) for layer in layers}


def build_steering_vectors_multi_internal_true(
    processor,
    model,
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    modalities: list[str],
    layers: list[int],
    calibration_samples: int,
    normalize: str,
) -> dict[int, torch.Tensor]:
    selected_rows = rows if calibration_samples <= 0 else rows[:calibration_samples]
    if not selected_rows:
        raise ValueError("Need at least one calibration row.")

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {layer: 0 for layer in layers}
    for row in tqdm(selected_rows, desc="building text steering vectors"):
        for modality in modalities:
            text_hidden = last_prompt_hidden_multi_internal_true(
                processor,
                model,
                row,
                image_root=image_root,
                modality=modality,
                instruction=TEXT_STEER_INSTRUCTION,
            )
            image_hidden = last_prompt_hidden_multi_internal_true(
                processor,
                model,
                row,
                image_root=image_root,
                modality=modality,
                instruction=IMAGE_STEER_INSTRUCTION,
            )
            for layer in layers:
                hidden_index = layer + 1
                if hidden_index >= len(text_hidden):
                    raise IndexError(f"Layer {layer} is out of range for {len(text_hidden) - 1} decoder layers.")
                delta = (text_hidden[hidden_index] - image_hidden[hidden_index]).mean(dim=0)
                sums[layer] = delta if layer not in sums else sums[layer] + delta
                counts[layer] += 1

    return {layer: normalize_vector(sums[layer] / max(counts[layer], 1), normalize) for layer in layers}


def is_layer_vector_dict(vectors: Any) -> bool:
    return isinstance(vectors, dict) and all(isinstance(key, int) for key in vectors)


def vectors_for_order(vectors: Any, order: str) -> dict[int, torch.Tensor]:
    if is_layer_vector_dict(vectors):
        return vectors
    if not isinstance(vectors, dict):
        raise TypeError(f"Unexpected vectors type: {type(vectors)}")
    selected = vectors.get(order) or vectors.get(base_order(order))
    if selected is None:
        available = ", ".join(str(key) for key in vectors)
        raise KeyError(f"No steering vector for order={order}. Available: {available}")
    return selected


def vector_layer_keys(vectors: Any) -> list[int]:
    if is_layer_vector_dict(vectors):
        return sorted(vectors)
    layers = set()
    for group_vectors in vectors.values():
        layers.update(int(layer) for layer in group_vectors)
    return sorted(layers)


def serialize_vectors(vectors: Any) -> Any:
    if is_layer_vector_dict(vectors):
        return {int(layer): vector.cpu() for layer, vector in vectors.items()}
    return {
        str(group): {int(layer): vector.cpu() for layer, vector in group_vectors.items()}
        for group, group_vectors in vectors.items()
    }


def build_behavior_steering_vectors_multi_internal_true(
    processor,
    model,
    rows: list[dict[str, Any]],
    *,
    row_indices: list[int] | None,
    image_root: Path,
    modalities: list[str],
    layers: list[int],
    calibration_samples: int,
    normalize: str,
    behavior_tags_path: Path,
    behavior_field_prefix: str,
) -> dict[str, dict[int, torch.Tensor]]:
    tagged_payload = json.loads(behavior_tags_path.read_text(encoding="utf-8"))
    tagged_rows = tagged_payload.get("rows") if isinstance(tagged_payload, dict) else None
    if not isinstance(tagged_rows, list):
        raise TypeError(f"Expected tag JSON with a rows list: {behavior_tags_path}")
    if len(tagged_rows) < len(rows):
        raise ValueError(f"Tag rows({len(tagged_rows)}) are fewer than data rows({len(rows)}).")
    if row_indices is None:
        row_indices = list(range(len(rows)))
    if len(row_indices) != len(rows):
        raise ValueError(f"row_indices({len(row_indices)}) must match rows({len(rows)}).")
    if tagged_rows and max(row_indices, default=-1) >= len(tagged_rows):
        raise ValueError(f"Tag rows({len(tagged_rows)}) do not cover requested calibration row indices.")

    limit = len(rows) if calibration_samples <= 0 else min(calibration_samples, len(rows))
    indexed_rows = list(zip(row_indices, rows))[:limit]
    prefix = behavior_field_prefix.strip().rstrip("-")
    groups = sorted({base_order(modality) for modality in modalities})
    pos_sums: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}
    neg_sums: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}
    pos_counts: dict[str, int] = {group: 0 for group in groups}
    neg_counts: dict[str, int] = {group: 0 for group in groups}

    for source_row_idx, row in tqdm(indexed_rows, desc="building behavior steering vectors"):
        tagged = tagged_rows[source_row_idx]
        for modality in modalities:
            group = base_order(modality)
            field = f"{prefix}-{modality}" if prefix else modality
            tag = tagged.get(f"{field}-final_tag")
            if tag not in {"Text", "Image"}:
                continue

            hidden = last_prompt_hidden_multi_internal_true(
                processor,
                model,
                row,
                image_root=image_root,
                modality=modality,
                instruction=None,
            )
            target_sums = pos_sums if tag == "Text" else neg_sums
            target_counts = pos_counts if tag == "Text" else neg_counts
            for layer in layers:
                hidden_index = layer + 1
                if hidden_index >= len(hidden):
                    raise IndexError(f"Layer {layer} is out of range for {len(hidden) - 1} decoder layers.")
                value = hidden[hidden_index].mean(dim=0)
                bucket = target_sums[group]
                bucket[layer] = value if layer not in bucket else bucket[layer] + value
            target_counts[group] += 1

    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for group in groups:
        if pos_counts[group] == 0 or neg_counts[group] == 0:
            raise ValueError(
                f"Need both Text and Image calibration examples for {group}; "
                f"got Text={pos_counts[group]} Image={neg_counts[group]}."
            )
        vectors[group] = {}
        for layer in layers:
            positive = pos_sums[group][layer] / pos_counts[group]
            negative = neg_sums[group][layer] / neg_counts[group]
            vectors[group][layer] = normalize_vector(positive - negative, normalize)
        print(f"[behavior-vector] {group}: Text={pos_counts[group]} Image={neg_counts[group]}")
    return vectors


def save_vectors(path: Path, vectors: Any, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": serialize_vectors(vectors),
            "layers": vector_layer_keys(vectors),
            "vector_source": args.vector_source,
            "behavior_tags_path": args.behavior_tags_path,
            "behavior_field_prefix": args.behavior_field_prefix,
            "scale": args.steer_scale,
            "order_scales": args.order_steer_scale_map,
            "normalize": args.normalize_steering,
            "task": args.task,
            "text_field": args.text_field,
            "image_kind": args.image_kind,
            "calibration_samples": args.calibration_samples,
            "calibration_orders": args.calibration_orders,
            "calibration_start": getattr(args, "calibration_start", None),
            "calibration_end": getattr(args, "calibration_end", None),
            "calibration_row_count": getattr(args, "calibration_row_count", None),
            "eval_start": getattr(args, "eval_start", None),
            "eval_end": getattr(args, "eval_end", None),
            "eval_row_count": getattr(args, "eval_row_count", None),
        },
        path,
    )


def load_vectors(path: Path) -> Any:
    payload = torch.load(path, map_location="cpu")
    vectors = payload["vectors"] if isinstance(payload, dict) and "vectors" in payload else payload
    if all(str(key).lstrip("-").isdigit() for key in vectors):
        return {int(layer): vector.detach().cpu() for layer, vector in vectors.items()}
    return {
        str(group): {int(layer): vector.detach().cpu() for layer, vector in group_vectors.items()}
        for group, group_vectors in vectors.items()
    }


def add_vector_to_output(output: Any, vector: torch.Tensor, scale: float, all_tokens: bool) -> Any:
    if isinstance(output, tuple):
        hidden = output[0]
        rest = output[1:]
    else:
        hidden = output
        rest = None

    if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[-1] != vector.numel():
        return output

    steer = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1) * scale
    if all_tokens:
        hidden = hidden + steer
    else:
        hidden = hidden.clone()
        hidden[:, -1:, :] = hidden[:, -1:, :] + steer
    return (hidden, *rest) if rest is not None else hidden


@contextmanager
def steering_hooks(model, vectors: dict[int, torch.Tensor], scale: float, all_tokens: bool):
    layers = get_decoder_layers(model)
    handles = []
    for layer_idx, vector in vectors.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} out of range; model has {len(layers)} layers.")

        def hook(_module, _inputs, output, *, steer_vector=vector):
            return add_vector_to_output(output, steer_vector, scale, all_tokens)

        handles.append(layers[layer_idx].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_one(processor, model, row: dict[str, Any], args: argparse.Namespace, order: str) -> str:
    image = Image.open(image_path_for(row, Path(args.image_root), args.image_kind)).convert("RGB")
    prompt = prompt_for(row, args.text_field)
    inputs = encode_inputs(processor, model, prompt, image, order)
    do_sample = args.temperature > 0
    generation_kwargs = {"do_sample": do_sample, "max_new_tokens": args.max_new_tokens}
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    input_len = inputs.input_ids.shape[-1]
    trimmed = generated_ids[:, input_len:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip().strip(".")


def generate_one_multi_internal_true(
    processor,
    model,
    row: dict[str, Any],
    args: argparse.Namespace,
    modality: str,
) -> str:
    image = Image.open(multi_internal_true_image_path(row, Path(args.image_root), modality)).convert("RGB")
    prompt = multi_internal_true_prompt(row, modality)
    inputs = encode_inputs(processor, model, prompt, image, modality)
    do_sample = args.temperature > 0
    generation_kwargs = {"do_sample": do_sample, "max_new_tokens": args.max_new_tokens}
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    input_len = inputs.input_ids.shape[-1]
    trimmed = generated_ids[:, input_len:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip().strip(".")


def generate_n(
    processor,
    model,
    row: dict[str, Any],
    args: argparse.Namespace,
    order: str,
    *,
    vectors: dict[int, torch.Tensor] | None,
) -> list[str]:
    answers = []
    scale = scale_for_order(order, args)
    for _ in range(args.num_samples):
        if vectors:
            order_vectors = vectors_for_order(vectors, order)
            with steering_hooks(model, order_vectors, scale, args.steer_all_tokens):
                answers.append(generate_one(processor, model, row, args, order))
        else:
            answers.append(generate_one(processor, model, row, args, order))
    return answers


def generate_n_multi_internal_true(
    processor,
    model,
    row: dict[str, Any],
    args: argparse.Namespace,
    modality: str,
    *,
    vectors: dict[int, torch.Tensor] | None,
) -> list[str]:
    answers = []
    scale = scale_for_order(modality, args)
    for _ in range(args.num_samples):
        if vectors:
            order_vectors = vectors_for_order(vectors, modality)
            with steering_hooks(model, order_vectors, scale, args.steer_all_tokens):
                answers.append(generate_one_multi_internal_true(processor, model, row, args, modality))
        else:
            answers.append(generate_one_multi_internal_true(processor, model, row, args, modality))
    return answers


def load_model_and_processor(args: argparse.Namespace):
    processor_source = args.processor_name_or_path or args.adapter_path or args.model_name_or_path
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, processor


def apply_task_preset(args: argparse.Namespace) -> None:
    preset = TASK_PRESETS[args.task]
    if args.text_field is None:
        args.text_field = preset["text_field"]
    if args.image_kind is None:
        args.image_kind = preset["image_kind"]
    if args.data_path is None and preset.get("data_path"):
        args.data_path = preset["data_path"]
    if args.image_root == parser_default_image_root() and preset.get("image_root"):
        args.image_root = preset["image_root"]
    if args.orders is None and preset.get("orders"):
        args.orders = preset["orders"]
    if args.calibration_orders is None and preset.get("calibration_orders"):
        args.calibration_orders = preset["calibration_orders"]
    if args.output_file is None:
        args.output_file = preset["output_file"]


def existing_or_default_data_path(args: argparse.Namespace) -> Path:
    if args.data_path:
        return Path(args.data_path)
    preferred = Path("./data/MMKC/ConflictQA/data/qwen7b/mcq_internal=false_para.json")
    legacy = Path("./data/MMKC/ConflictQA/data/mcq/qwen7b/mcq_internal=false_para.json")
    return preferred if preferred.exists() else legacy


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--image_root", default=parser_default_image_root())
    parser.add_argument("--task", choices=sorted(TASK_PRESETS), default="multi_text_true_para")
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--image_kind", choices=["para", "counter", "row"], default=None)
    parser.add_argument("--orders", default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--calibration_samples", type=int, default=32)
    parser.add_argument("--calibration_start", type=int, default=0)
    parser.add_argument("--calibration_end", type=int, default=None)
    parser.add_argument("--eval_start", type=int, default=None)
    parser.add_argument("--eval_end", type=int, default=None)
    parser.add_argument("--calibration_orders", default=None)
    parser.add_argument("--layers", default="20-23")
    parser.add_argument("--steer_scale", type=float, default=1.0)
    parser.add_argument("--order_steer_scales", default=None, help="Optional comma-separated per-order scales, e.g. itq=-5,tiq=5.")
    parser.add_argument("--normalize_steering", choices=["l2", "rms", "none"], default="l2")
    parser.add_argument("--vector_source", choices=["instruction", "behavior"], default="instruction")
    parser.add_argument("--behavior_tags_path", default=None)
    parser.add_argument("--behavior_field_prefix", default="baseline-output")
    parser.add_argument("--steer_all_tokens", action="store_true")
    parser.add_argument("--vector_path", default=None)
    parser.add_argument("--save_vector_path", default=None)
    parser.add_argument("--also_baseline", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()
    apply_task_preset(args)
    if args.orders is None:
        args.orders = "itq,tiq"
    if args.calibration_orders is None:
        args.calibration_orders = "itq,tiq"
    args.order_steer_scale_map = parse_order_scales(args.order_steer_scales)
    import_runtime_deps()

    data_path = existing_or_default_data_path(args)
    rows = read_json(data_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    calibration_indices, calibration_rows = slice_rows_with_indices(
        rows, start=args.calibration_start, end=args.calibration_end, label="calibration"
    )
    eval_indices, eval_rows = slice_rows_with_indices(
        rows, start=args.eval_start, end=args.eval_end, label="eval"
    )
    if not calibration_rows:
        raise ValueError("Calibration split is empty.")
    if not eval_rows:
        raise ValueError("Evaluation split is empty.")
    args.calibration_row_count = len(calibration_rows)
    args.eval_row_count = len(eval_rows)
    print(
        f"Split: calibration rows={len(calibration_rows)} "
        f"[{calibration_indices[0]}:{calibration_indices[-1] + 1}], "
        f"eval rows={len(eval_rows)} [{eval_indices[0]}:{eval_indices[-1] + 1}]"
    )

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    model, processor = load_model_and_processor(args)
    layers = parse_layers(args.layers)
    if args.vector_path:
        vectors = load_vectors(Path(args.vector_path))
    elif args.task == "multi_internal_true_para" and args.vector_source == "behavior":
        if not args.behavior_tags_path:
            raise ValueError("--vector_source behavior requires --behavior_tags_path")
        vectors = build_behavior_steering_vectors_multi_internal_true(
            processor,
            model,
            calibration_rows,
            row_indices=calibration_indices,
            image_root=Path(args.image_root),
            modalities=[item.strip() for item in args.calibration_orders.split(",") if item.strip()],
            layers=layers,
            calibration_samples=args.calibration_samples,
            normalize=args.normalize_steering,
            behavior_tags_path=Path(args.behavior_tags_path),
            behavior_field_prefix=args.behavior_field_prefix,
        )
    elif args.task == "multi_internal_true_para":
        vectors = build_steering_vectors_multi_internal_true(
            processor,
            model,
            calibration_rows,
            image_root=Path(args.image_root),
            modalities=[item.strip() for item in args.calibration_orders.split(",") if item.strip()],
            layers=layers,
            calibration_samples=args.calibration_samples,
            normalize=args.normalize_steering,
        )
    else:
        vectors = build_steering_vectors(
            processor,
            model,
            calibration_rows,
            text_field=args.text_field,
            image_root=Path(args.image_root),
            image_kind=args.image_kind,
            orders=[item.strip() for item in args.calibration_orders.split(",") if item.strip()],
            layers=layers,
            calibration_samples=args.calibration_samples,
            normalize=args.normalize_steering,
        )

    vector_path = Path(args.save_vector_path) if args.save_vector_path else output_dir / "text_steering_vectors.pt"
    save_vectors(vector_path, vectors, args)

    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    out_rows = []
    for source_row_idx, row in tqdm(list(zip(eval_indices, eval_rows)), desc="generating steered ConflictQA multi"):
        item = dict(row)
        item["_source_row_idx"] = source_row_idx
        for order in orders:
            if args.task == "multi_internal_true_para":
                if args.also_baseline:
                    item[f"baseline-output-{order}"] = generate_n_multi_internal_true(
                        processor, model, row, args, order, vectors=None
                    )
                item[f"output-{order}"] = generate_n_multi_internal_true(
                    processor, model, row, args, order, vectors=vectors
                )
            else:
                if args.also_baseline:
                    item[f"baseline-output-{order}"] = generate_n(processor, model, row, args, order, vectors=None)
                item[f"output-{order}"] = generate_n(processor, model, row, args, order, vectors=vectors)
        item["_text_steering"] = {
            "layers": vector_layer_keys(vectors),
            "scale": args.steer_scale,
            "order_scales": args.order_steer_scale_map,
            "normalize": args.normalize_steering,
            "vector_path": str(vector_path),
            "task": args.task,
            "text_field": args.text_field,
            "image_kind": args.image_kind,
            "calibration_start": args.calibration_start,
            "calibration_end": args.calibration_end,
            "calibration_row_count": args.calibration_row_count,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "eval_row_count": args.eval_row_count,
            "source_row_idx": source_row_idx,
        }
        out_rows.append(item)

    output_file.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_file)
    print(f"Saved steering vectors to {vector_path}")


if __name__ == "__main__":
    main()
