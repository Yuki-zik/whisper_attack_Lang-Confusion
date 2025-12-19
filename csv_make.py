"""
生成 LibriSpeech 分割集的 CSV 清单，便于 run_attack / fit_attacker 使用。

新增能力：
- 支持通过 --role 显式指定生成 fit.csv / test-clean.csv
- 若未指定 --role，则保持原行为（使用 split 目录名）

使用示例：
    # 生成训练用 fit.csv
    python tools/generate_librispeech_csv.py \
        --split-path /root/.../LibriSpeech/train-clean-100 \
        --output-dir /root/.../LibriSpeech/csv \
        --role fit \
        --lang en --compute-duration

    # 生成评估用 test-clean.csv
    python tools/generate_librispeech_csv.py \
        --split-path /root/.../LibriSpeech/test-clean \
        --output-dir /root/.../LibriSpeech/csv \
        --role testclean \
        --lang en --compute-duration
"""

import argparse
import csv
import pathlib
import soundfile


def load_transcripts(split_root: pathlib.Path) -> dict:
    """读取 LibriSpeech 分割集中的 .trans.txt 文件，返回 {音频ID: 文本}。"""
    transcripts = {}
    for trans_file in split_root.rglob("*.trans.txt"):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(" ", maxsplit=1)
            audio_id = parts[0]
            text = parts[1] if len(parts) == 2 else ""
            transcripts[audio_id] = text
    return transcripts


def collect_rows(
    split_root: pathlib.Path,
    transcripts: dict,
    lang: str,
    compute_duration: bool,
) -> list:
    """收集当前分割集的行数据。"""
    rows = []
    for wav in sorted(split_root.rglob("*.flac")):
        audio_id = wav.stem
        duration = 0.0
        if compute_duration:
            info = soundfile.info(str(wav))
            duration = info.frames / info.samplerate

        row = {
            "ID": audio_id,
            "duration": duration,
            "wav": str(wav),
            "wrd": transcripts.get(audio_id, ""),
        }
        if lang:
            row["lang"] = lang

        rows.append(row)
    return rows


def resolve_output_name(split_path: pathlib.Path, role: str | None) -> str:
    """
    根据 role 决定输出 CSV 名称：
    - role=fit        -> fit.csv
    - role=testclean  -> test-clean.csv
    - role=None       -> 使用 split_path.name.csv（保持原行为）
    """
    if role is None:
        return f"{split_path.name}.csv"

    role = role.lower()
    if role == "fit":
        return "fit.csv"
    if role in ("test", "testclean", "test-clean"):
        return "test-clean.csv"

    raise ValueError(f"未知 role: {role}（支持 fit / testclean）")


def generate_csv(
    split_path: pathlib.Path,
    output_dir: pathlib.Path,
    lang: str,
    compute_duration: bool,
    role: str | None,
) -> pathlib.Path:
    """为单个分割集生成 CSV 并返回输出路径。"""
    if not split_path.exists():
        raise FileNotFoundError(f"分割集路径不存在: {split_path}")

    transcripts = load_transcripts(split_path)
    rows = collect_rows(split_path, transcripts, lang, compute_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = resolve_output_name(split_path, role)
    output_path = output_dir / csv_name

    fieldnames = ["ID", "duration", "wav", "wrd"]
    if lang:
        fieldnames.append("lang")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 LibriSpeech CSV")
    parser.add_argument(
        "--split-path",
        required=True,
        action="append",
        type=pathlib.Path,
        help="单个分割集的根目录（可重复指定多个）",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="CSV 输出目录，默认使用第一个 split-path 的父目录下的 csv 子目录",
    )
    parser.add_argument(
        "--role",
        choices=["fit", "testclean"],
        default=None,
        help="可选：显式指定 CSV 用途（fit 或 testclean）",
    )
    parser.add_argument(
        "--lang",
        default="",
        help="可选，固定写入 lang 列的语言代码（如 en、es）",
    )
    parser.add_argument(
        "--compute-duration",
        action="store_true",
        help="若提供则读取音频计算 duration；未提供时填充 0 以加快扫描",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.split_path[0].parent / "csv"

    for split_path in args.split_path:
        output_path = generate_csv(
            split_path=split_path,
            output_dir=output_dir,
            lang=args.lang,
            compute_duration=args.compute_duration,
            role=args.role,
        )
        num_rows = sum(1 for _ in output_path.open(encoding="utf-8")) - 1
        print(f"已生成: {output_path} (共 {num_rows} 条)")


if __name__ == "__main__":
    main()
