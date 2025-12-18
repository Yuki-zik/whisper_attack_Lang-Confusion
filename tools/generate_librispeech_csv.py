"""生成 LibriSpeech 分割集的 CSV 清单，便于 run_attack/fit_attacker 使用。

使用示例：
    python tools/generate_librispeech_csv.py \
        --split-path /root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/dev-clean \
        --output-dir /root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/csv \
        --lang en --compute-duration

生成的 CSV 默认包含列 ID、duration、wav、wrd；
- 若提供 --lang，将额外生成 lang 列；
- 若未提供文字转写，则 wrd 为空字符串。
"""

import argparse
import csv
import pathlib
import soundfile


def load_transcripts(split_root: pathlib.Path) -> dict:
    """读取 LibriSpeech 分割集中的 .trans.txt 文件，返回 {音频ID: 文本}。

    LibriSpeech 的转写文件以每行 "<音频ID> <文本>" 的形式存储。
    """

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


def collect_rows(split_root: pathlib.Path, transcripts: dict, lang: str, compute_duration: bool) -> list:
    """收集当前分割集的行数据。"""

    rows = []
    for wav in sorted(split_root.rglob("*.flac")):
        audio_id = wav.stem
        duration = 0.0
        if compute_duration:
            info = soundfile.info(str(wav))
            duration = info.frames / info.samplerate
        rows.append(
            {
                "ID": audio_id,
                "duration": duration,
                "wav": str(wav),
                "wrd": transcripts.get(audio_id, ""),
                **({"lang": lang} if lang else {}),
            }
        )
    return rows


def generate_csv(split_path: pathlib.Path, output_dir: pathlib.Path, lang: str, compute_duration: bool) -> pathlib.Path:
    """为单个分割集生成 CSV 并返回输出路径。"""

    if not split_path.exists():
        raise FileNotFoundError(f"分割集路徑不存在: {split_path}")

    transcripts = load_transcripts(split_path)
    rows = collect_rows(split_path, transcripts, lang, compute_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_path.name}.csv"

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
        help="CSV 输出目录，默认使用第一個分割集的父目錄下的 csv 子目錄",
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
        output_path = generate_csv(split_path, output_dir, args.lang, args.compute_duration)
        print(f"已生成: {output_path} (共 {sum(1 for _ in output_path.open()) - 1} 條)")


if __name__ == "__main__":
    main()
