"""生成動画の発話を音声認識にかけ、**実際に日本語で喋っているか**を客観的に判定する。

背景: H3 の公式プロンプト仕様は台詞を `<d>[Japanese] …</d>` で囲み逐語保持することを
求めるが、**そう書けば本当に日本語で発話されるのか**はこのリポジトリで未検証だった
(記法が公式どおりであることと、生成音声が日本語であることは別の話)。目視ならぬ
「耳で確認」ができないので、ASR に判定させる。

判定に使うのは faster-whisper。この venv には無いため、`easy_music_v2/.venv` の
インタプリタで**別プロセスとして**呼ぶ(本アプリの venv を汚さない)。

出力: 検出言語とその確率、認識されたテキスト、期待した台詞との一致度。
`--expect` に台詞を渡すと、文字単位の一致率も出す。

実行:
  venv/bin/python scripts/probe_speech_language.py outputs/t2va_xxx.mp4 \
      --expect "今日はいい天気だね。"
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# faster-whisper を持つ別プロジェクトの venv(このアプリの venv には入れない)
ASR_PYTHON = Path("/home/animede/easy_music_v2/.venv/bin/python")

_ASR_SNIPPET = r"""
import json, sys
from faster_whisper import WhisperModel
wav, model_size = sys.argv[1], sys.argv[2]
m = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = m.transcribe(wav, beam_size=5)
segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in segments]
print(json.dumps({
    "language": info.language,
    "language_probability": round(info.language_probability, 4),
    "duration": round(info.duration, 2),
    "segments": segs,
    "text": "".join(s["text"] for s in segs).strip(),
}, ensure_ascii=False))
"""


def extract_wav(media: Path, out_wav: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(media),
         "-vn", "-ac", "1", "-ar", "16000", str(out_wav)],
        check=True,
    )


def char_overlap(a: str, b: str) -> float:
    """期待した台詞と認識結果の、文字集合ベースの粗い一致率(句読点は無視)。"""
    strip = str.maketrans("", "", "、。！？ 　,.!?")
    sa, sb = set(a.translate(strip)), set(b.translate(strip))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media", nargs="+", help="mp4/wav のパス")
    ap.add_argument("--expect", default=None, help="期待する台詞(文字一致率を出す)")
    ap.add_argument("--model", default="small", help="whisper モデルサイズ")
    args = ap.parse_args()

    if not ASR_PYTHON.exists():
        raise SystemExit(f"ASR 用の python が見つかりません: {ASR_PYTHON}")

    results = []
    for path in args.media:
        media = Path(path)
        if not media.exists():
            print(f"[skip] {media} が存在しません")
            continue
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            extract_wav(media, wav)
            proc = subprocess.run(
                [str(ASR_PYTHON), "-c", _ASR_SNIPPET, str(wav), args.model],
                capture_output=True, text=True,
            )
        if proc.returncode != 0:
            print(f"[fail] {media.name}: {proc.stderr.strip()[-400:]}")
            continue
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        rec["file"] = media.name
        if args.expect:
            rec["expected"] = args.expect
            rec["char_overlap"] = round(char_overlap(args.expect, rec["text"]), 3)
        results.append(rec)

        print(f"=== {media.name} ===")
        print(f"  検出言語 : {rec['language']} (確率 {rec['language_probability']})")
        print(f"  認識結果 : {rec['text']!r}")
        if args.expect:
            print(f"  期待台詞 : {args.expect!r}")
            print(f"  文字一致率: {rec['char_overlap']:.1%}")
        for s in rec["segments"]:
            print(f"    [{s['start']:>5.2f}-{s['end']:>5.2f}] {s['text']}")
    print()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
