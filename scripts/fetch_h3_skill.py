# -*- coding: utf-8 -*-
"""MiniMax公式 `h3-prompt-writing` スキル(SKILL.md + references/*.txt)を取得し、
`skills_cache/` へ保存する。

背景: `MiniMax-AI/MiniMax-H3` リポジトリはライセンス表記が無い(GitHub API で
`license: null`)ため、内容をこのリポジトリへ同梱・再配布しない方針にした。
`core/llm.py` の "h3-official" モードは、このスクリプトで事前に取得済みの
`skills_cache/` を読んで動作する(未取得なら明確な日本語エラーで案内する)。

再実行可能: 既存ファイルは上書きする。ネットワーク不可時は分かりやすいエラーで終了する。

使い方:
    venv/bin/python scripts/fetch_h3_skill.py
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/"
)

FILES = {
    "SKILL.md": "SKILL.md",
    "base-en.txt": "references/base-en.txt",
    "ref-en.txt": "references/ref-en.txt",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "skills_cache" / "h3-prompt-writing"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for local_name, rel_path in FILES.items():
        url = REPO_RAW_BASE + rel_path
        dest = OUT_DIR / local_name
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"[FAIL] {url} -> {e}", file=sys.stderr)
            ok = False
            continue
        dest.write_bytes(data)
        print(f"[OK] {rel_path} -> {dest} ({len(data)} bytes)")
    if not ok:
        print(
            "一部のファイル取得に失敗しました。ネットワーク接続を確認し再実行してください。",
            file=sys.stderr,
        )
        return 1
    print(f"完了: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
