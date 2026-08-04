# -*- coding: utf-8 -*-
"""ローカルLLM(gemma4-31B等、OpenAI互換 /v1/chat/completions)による H3 向けプロンプト強化。

背景(dev_notes/handoff-minimax-h3.md「プロンプトとOmniの関係」):
H3 のクラウド版(Hailuo AI)は内部にプロンプト整形層を持つが、オープンウェイト版には無い。
本モジュールはローカルLLMでその整形層を再現する。H3 の公式推奨構造は再生順ブリーフ
「シーン→被写体→アクション→カメラ→音の意図→終わり方」(上限7,000字)。

実機検証済みの重要事実(2026-08-04、このワークスペースの検証):
- H3 はマルチショットをネイティブ対応。`CUT n [X-Y秒]: ...` 形式のタイムコードブロックで
  1クリップ内にハードカットを実行できる(実測: 10秒・2カット指定で6.2秒地点にカット、
  タイムコード精度は±1秒程度)
- 日本語プロンプトがそのまま効く。焦点距離(35/50/65/100mm)・カメラワーク・音の指示も
  ショット単位で通る

接続先は環境変数 H3_LLM_URL(既定 http://127.0.0.1:64650)。gemma4-31B(Q4_K_M)で
動作確認済み。小型〜中型LLMは指示だけでは形式に従わないことがあるため、各モードに
few-shot 例を入れてある(diffusers-server の LLM強化で確立した知見)。
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_LLM_URL = "http://127.0.0.1:64650"
LLM_TIMEOUT_S = 120  # gemma4-31B Q4 は応答が遅めのため長めに


class LLMConnectionError(Exception):
    """LLMサーバに接続できない、またはLLMサーバがエラーを返した場合。"""


def get_llm_url() -> str:
    return os.environ.get("H3_LLM_URL", DEFAULT_LLM_URL).rstrip("/")


def chat_completion(system_prompt: str, user_text: str, *, temperature: float = 0.6) -> str:
    url = f"{get_llm_url()}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise LLMConnectionError(f"LLMサーバ({get_llm_url()})に接続できません: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMConnectionError(f"LLMサーバの応答を解釈できません: {e}") from e
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise LLMConnectionError(f"LLMサーバの応答形式が想定外です: {data}") from e


# ---------------------------------------------------------------------------
# モード別 system prompt(few-shot 付き)
# ---------------------------------------------------------------------------

_COMMON_RULES = (
    "あなたは動画生成モデル MiniMax H3 (Hailuo 3.0) 専用のプロンプトエンジニアです。"
    "H3 は動画とステレオ音声を同時生成するため、映像だけでなく音の指示も重要です。"
    "ユーザーの意図・被写体・指定済みの要素は変えないこと。"
    "出力はプロンプト本文のみ。前置き・説明・引用符・「プロンプト:」等のラベルは一切禁止。"
)

BRIEF_SYSTEM_PROMPT = (
    _COMMON_RULES
    + "ユーザーの短い入力を、H3公式推奨の再生順ブリーフ形式の日本語プロンプトに詳細化してください。"
    "構造は必ず「シーン(場所・時間帯・光)→被写体(外見の具体描写)→アクション(時間順の動き)"
    "→カメラ(ショットサイズ・焦点距離は35mm/50mm/65mm/100mmのみ使用可・カメラワーク)→音の意図"
    "(環境音・効果音・雰囲気)→終わり方(最後の1〜2秒の画)」の一段落構成。"
    "ユーザーが指定していない固有名詞や新しい登場人物を勝手に追加しないこと。\n\n"
    "例:\n"
    "入力: 雨の夜の交差点を歩く女性\n"
    "出力: 夜の都市の交差点、雨に濡れた路面がネオンを反射している。黒いコートに透明の傘を差した"
    "長髪の女性が、横断歩道を画面左から右へゆっくり歩いて渡る。ミディアムショット、50mm、"
    "女性の歩みに合わせた緩やかなトラッキング。雨音と遠くの車の走行音、傘に当たる雨粒の音が重なる。"
    "最後は渡り終えた女性が立ち止まり、信号の光が路面に滲んで終わる。"
)

STORYBOARD_SYSTEM_PROMPT_TEMPLATE = (
    _COMMON_RULES
    + "ユーザーの入力を、H3のマルチショット(カット割り)形式の日本語プロンプトに展開してください。"
    "総尺は{seconds}秒。2〜3個のカットに分割し、各カットを"
    "「CUT n [開始-終了秒]: シーンと被写体/アクション。カメラ(ショットサイズ・焦点距離)。音。」"
    "の形式で書くこと。焦点距離は35mm/50mm/65mm/100mmの4種のみ使用可(それ以外の値は禁止)。カット間は絵が明確に変わるハードカットとし、場面・時間帯・カメラの少なくとも"
    "1つを大きく変化させ、音もカットに合わせて変化させること。被写体の同一性(同じ人物・同じ動物)は"
    "カットをまたいで維持する指示を入れること。タイムコードの合計は必ず{seconds}秒に一致させること。\n\n"
    "例(総尺10秒の場合):\n"
    "入力: 商店街の猫\n"
    "出力: CUT 1 [0-5秒]: 昼間の日本の商店街。三毛猫が魚屋の店先に座って魚を見つめている。"
    "ロングショット、35mm、固定。商店街の雑踏と呼び込みの声。\n"
    "CUT 2 [5-10秒]: ハードカットで夜の同じ商店街。シャッターが閉まり、同じ三毛猫が街灯の下を"
    "歩いている。ローアングルのクローズアップ、100mm、猫を追うゆっくりとしたトラッキング。"
    "静かな夜、遠くの虫の音と猫の足音。"
)

TRANSLATE_SYSTEM_PROMPT = (
    _COMMON_RULES
    + "ユーザーの日本語入力を、動画生成プロンプトとして自然な英語に翻訳してください。"
    "意訳・詳細の追加・省略はせず、原文の内容を忠実に英語化すること。"
    "CUT n [X-Ys]: のようなタイムコード構造がある場合は構造をそのまま保つこと。\n\n"
    "例:\n"
    "入力: 夕暮れの海辺を歩く銀髪の少女、波の音\n"
    "出力: A silver-haired girl walking along the seashore at dusk, with the sound of waves."
)

VALID_MODES = ("brief", "storyboard", "translate")


def enhance_prompt(text: str, mode: str, seconds: float = 10.0) -> str:
    if mode == "brief":
        return chat_completion(BRIEF_SYSTEM_PROMPT, text)
    if mode == "storyboard":
        sec = int(round(seconds))
        return chat_completion(STORYBOARD_SYSTEM_PROMPT_TEMPLATE.format(seconds=sec), text)
    if mode == "translate":
        return chat_completion(TRANSLATE_SYSTEM_PROMPT, text, temperature=0.2)
    raise ValueError(f"mode は {VALID_MODES} のいずれかです: {mode!r}")
