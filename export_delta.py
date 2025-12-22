#!/usr/bin/env python3
"""
从 delta/delta.ckpt 等检查点中提取通用扰动，并自动命名后保存为 .npy 文件。

用法示例：
python export_delta.py \
    --checkpoint /path/to/attacks/univ_lang/es/whisper-tiny/1002/params/delta.ckpt \
    --output-dir ./exported
    # 若未指定 --name，会自动根据路径生成形如 delta_univ_lang_es_whisper-tiny_1002.npy 的文件名。
"""

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


TensorWithPath = Tuple[torch.Tensor, Tuple[str, ...]]


def _collect_tensor_paths(obj: Any, prefix: Tuple[str, ...] = ()) -> List[TensorWithPath]:
    """收集 checkpoint 中所有张量及其路径，便于给出可选项提示。"""

    results: List[TensorWithPath] = []

    if torch.is_tensor(obj):
        results.append((obj, prefix))
        return results

    if isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_collect_tensor_paths(v, prefix + (str(k),)))
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            results.extend(_collect_tensor_paths(v, prefix + (str(idx),)))

    return results


def _find_first_tensor(
    obj: Any, preferred_keys: Iterable[str]
) -> Optional[TensorWithPath]:
    """递归搜索 checkpoint 对象中的张量。

    先按 preferred_keys 尝试匹配常见字段（如 delta/tensor/univ_perturb），
    再回退到深度优先搜索，确保能提取出第一块张量。
    """

    def _search(value: Any, path: Tuple[str, ...]) -> Optional[TensorWithPath]:
        if torch.is_tensor(value):
            return value, path
        if isinstance(value, dict):
            for k in preferred_keys:
                if k in value:
                    candidate = _search(value[k], path + (str(k),))
                    if candidate is not None:
                        return candidate
            for k, v in value.items():
                candidate = _search(v, path + (str(k),))
                if candidate is not None:
                    return candidate
        elif isinstance(value, (list, tuple)):
            for idx, v in enumerate(value):
                candidate = _search(v, path + (str(idx),))
                if candidate is not None:
                    return candidate
        return None

    return _search(obj, ())


def _follow_key_path(obj: Any, key_path: Sequence[str]) -> Optional[TensorWithPath]:
    """按给定路径精确提取张量，路径元素按字典键或序号依次索引。"""

    cur: Any = obj
    path: Tuple[str, ...] = tuple(key_path)

    for key in key_path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, (list, tuple)):
            try:
                idx = int(key)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            return None

    if torch.is_tensor(cur):
        return cur, path
    return None


def load_delta_tensor(
    ckpt_path: Path,
    preferred_keys: Iterable[str],
    key_path: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """从 checkpoint 中提取通用扰动张量，返回张量及其路径。"""

    ckpt = torch.load(ckpt_path, map_location="cpu")

    tensor_with_path: Optional[TensorWithPath] = None
    if key_path:
        tensor_with_path = _follow_key_path(ckpt, key_path)
        if tensor_with_path is None:
            all_paths = _collect_tensor_paths(ckpt)
            choices = ["/".join(p) for _, p in all_paths]
            raise RuntimeError(
                f"未在 {ckpt_path} 的路径 {'/'.join(key_path)} 找到张量。"
                f" 可选路径示例：{choices[:10]}"
            )

    if tensor_with_path is None:
        tensor_with_path = _find_first_tensor(ckpt, preferred_keys)

    if tensor_with_path is None:
        all_paths = _collect_tensor_paths(ckpt)
        choices = ["/".join(p) for _, p in all_paths]
        raise RuntimeError(
            f"在 {ckpt_path} 中未找到扰动张量，请确认保存格式。可发现的张量路径：{choices[:10]}"
        )

    tensor, path = tensor_with_path
    return tensor.detach().cpu(), path


def infer_name(ckpt_path: Path, tensor_key: str, key_path: Optional[Sequence[str]]) -> str:
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
    if key_path:
        name_bits.append("-".join(key_path))

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
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="精确键路径（如 state_dict/tensor），优先于自动搜索",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preferred_keys = [args.tensor_key, "tensor", "univ_perturb", "perturbation"]
    key_path = args.key_path.split("/") if args.key_path else None

    tensor, path = load_delta_tensor(args.checkpoint, preferred_keys, key_path)
    np_data = tensor.squeeze().numpy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = args.name
    if filename is None:
        filename = infer_name(args.checkpoint, args.tensor_key, key_path)
    elif not filename.endswith(".npy"):
        filename = f"{filename}.npy"

    output_path = output_dir / filename
    np.save(output_path, np_data)

    print(f"已保存扰动到 {output_path}")
    print(
        "源张量路径: /{}".format("/".join(path) if path else "(root)"),
        f"形状: {np_data.shape}, 范围: min={np_data.min():.6f}, max={np_data.max():.6f}",
    )


if __name__ == "__main__":
    main()
