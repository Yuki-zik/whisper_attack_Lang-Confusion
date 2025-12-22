#!/usr/bin/env python3
"""
从 delta/delta.ckpt 等检查点中提取通用扰动，并自动命名后保存为 .npy 文件。

用法示例：
python export_delta.py \
    --checkpoint /root/autodl-tmp/whisper_attack_Lang-Confusion/attacks/univ_lang/zh/whisper-small/1101/CKPT+2025-12-22+18-30-01+00/delta.ckpt \
    --output-dir ./exported
    # 若未指定 --name，会自动根据路径生成形如 delta_univ_lang_es_whisper-tiny_1002.npy 的文件名。
"""

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch


def _find_first_tensor(obj: Any, preferred_keys: Iterable[str]) -> Optional[torch.Tensor]:
    """递归搜索 checkpoint 对象中的张量。

    先按 preferred_keys 尝试匹配常见字段（如 delta/tensor/univ_perturb），
    再回退到深度优先搜索，确保能提取出第一块张量。
    """

    def _search(value: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            for k in preferred_keys:
                if k in value:
                    candidate = _search(value[k])
                    if candidate is not None:
                        return candidate
            for v in value.values():
                candidate = _search(v)
                if candidate is not None:
                    return candidate
        elif isinstance(value, (list, tuple)):
            for v in value:
                candidate = _search(v)
                if candidate is not None:
                    return candidate
        return None

    return _search(obj)


def load_delta_tensor(ckpt_path: Path, preferred_keys: Iterable[str]) -> torch.Tensor:
    """从 checkpoint 中提取通用扰动张量。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tensor = _find_first_tensor(ckpt, preferred_keys)
    if tensor is None:
        raise RuntimeError(f"在 {ckpt_path} 中未找到扰动张量，请确认保存格式。")
    return tensor.detach().cpu()


def infer_name(ckpt_path: Path, tensor_key: str) -> str:
    """根据路径自动生成保存文件名，保证可读且唯一性较高。"""
    parts = list(ckpt_path.parts)
    name_bits = [ckpt_path.stem]

    if "attacks" in parts:
        idx = parts.index("attacks")
        name_bits.extend(parts[idx + 1 : idx + 5])
    else:
        parent = ckpt_path.parent.name
        if parent:
            name_bits.append(parent)

    if tensor_key:
        name_bits.append(tensor_key)

    slug = "_".join(filter(None, name_bits))
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", slug)
    if not slug.endswith(".npy"):
        slug += ".npy"
    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 checkpoint 导出通用扰动为 .npy")
    parser.add_argument("--checkpoint", required=True, type=Path, help="delta.ckpt 路径")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="保存 .npy 的目录，自动创建"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="自定义文件名（可选，缺省则自动生成），可不带 .npy 后缀",
    )
    parser.add_argument(
        "--tensor-key",
        type=str,
        default="delta",
        help="优先查找的键名，默认 delta，可根据实际 checkpoint 调整",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preferred_keys = [args.tensor_key, "tensor", "univ_perturb", "perturbation"]

    tensor = load_delta_tensor(args.checkpoint, preferred_keys)
    np_data = tensor.squeeze().numpy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = args.name
    if filename is None:
        filename = infer_name(args.checkpoint, args.tensor_key)
    elif not filename.endswith(".npy"):
        filename = f"{filename}.npy"

    output_path = output_dir / filename
    np.save(output_path, np_data)

    print(f"已保存扰动到 {output_path}")
    print(f"形状: {np_data.shape}, 范围: min={np_data.min():.6f}, max={np_data.max():.6f}")


if __name__ == "__main__":
    main()
