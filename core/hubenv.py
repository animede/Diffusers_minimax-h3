"""huggingface_hub の環境変数を、huggingface_hub を import する前に整える。

Windows では開発者モードが OFF(または非管理者)だと symlink を作れない。huggingface_hub は
それを検出してコピーに切り替えるが、検出が「先に True を書いてから試す」順序なので、
snapshot_download の並列ダウンロードでは検出中の他スレッドが symlink を試みて
`OSError: [WinError 1314] クライアントは要求された特権を保有していません` で落ちる(HF キャッシュを
初めて作る PC で起きる)。Windows では既定で symlink を使わない設定にして回避する
(symlink 無しでも blob を snapshot 側へ移動するだけなのでディスクは増えない)。
明示的に `HF_HUB_DISABLE_SYMLINKS=0` を置いた場合はそのまま尊重する。

呼び出しは huggingface_hub / diffusers / transformers のどれよりも先に行うこと(定数は import 時に読まれる)。
"""
from __future__ import annotations

import os
import sys


def apply() -> None:
    if sys.platform != "win32":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


apply()
