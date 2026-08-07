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
from pathlib import Path

DEFAULT_LLM_URL = "http://127.0.0.1:64650"
LLM_TIMEOUT_S = 180  # h3-official は system prompt が長く(15.8KB/23.6KB)応答も遅いため延長

_SKILLS_CACHE_DIR = Path(__file__).resolve().parent.parent / "skills_cache" / "h3-prompt-writing"


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

VALID_MODES = ("brief", "storyboard", "translate", "h3-official")
VALID_H3_OFFICIAL_TASKS = ("t2va", "fl2va", "ref2va")
VALID_H3_OFFICIAL_LANGS = ("en", "ja")


class H3SkillNotFetchedError(Exception):
    """`scripts/fetch_h3_skill.py` 未実行で skills_cache/ が存在しない場合。"""


def _read_skill_file(name: str) -> str:
    path = _SKILLS_CACHE_DIR / name
    if not path.is_file():
        raise H3SkillNotFetchedError(
            f"公式スキルのリファレンス({path})が見つかりません。"
            "先に次のコマンドで取得してください: "
            "venv/bin/python scripts/fetch_h3_skill.py"
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# h3-official モード: MiniMax公式 `h3-prompt-writing` スキル(SKILL.md +
# references/base-en.txt または references/ref-en.txt)をそのままシステムプロンプトへ
# 全文投入する。要約・独自解釈は行わず、公式のフィールド名・順序・記法を厳守させる。
# ---------------------------------------------------------------------------

_H3_OFFICIAL_WRAPPER_EN = (
    "You are a prompt-rewriting assistant for the MiniMax H3 (Hailuo 3.0) video+audio "
    "generation model. Follow the skill instructions and reference guide given below "
    "EXACTLY: use the exact field names, section order, labels, and timing notation from "
    "the guide. Do not invent new field names and do not omit any required field.\n\n"
    "The target video duration is {seconds:.2f} seconds. Every shot cut time and the final "
    "reference-alignment timestamp (for I2VA/FL2VA/L2VA) must fall within this duration, "
    "and the last shot must end at or before {seconds:.2f} seconds.\n\n"
    "Write the rewrite sections in English exactly as the guide specifies, EXCEPT: preserve "
    "dialogue/lyrics inside <d> tags and any on-screen text in their original language, "
    "exactly as the guide's own rules already require.\n\n"
    "Output ONLY the rewritten prompt (the instruction line if applicable, followed by the "
    "required fields in order). No preamble, no explanation, no markdown code fences, no "
    "labels like \"Prompt:\".\n\n"
    "--- BEGIN SKILL INSTRUCTIONS ---\n{skill_md}\n--- END SKILL INSTRUCTIONS ---\n\n"
    "--- BEGIN REFERENCE GUIDE ({guide_name}) ---\n{guide_text}\n--- END REFERENCE GUIDE ---"
)

_H3_OFFICIAL_WRAPPER_JA = (
    "あなたは MiniMax H3 (Hailuo 3.0、動画+音声同時生成モデル)向けのプロンプト書き換え"
    "アシスタントです。以下に示すスキル手順とリファレンスガイドに厳密に従ってください: "
    "ガイドに書かれているフィールド名・セクション順序・ラベル・タイムコード記法を"
    "そのまま使うこと。新しいフィールド名を作らないこと、必須フィールドを省略しないこと。\n\n"
    "対象動画の尺は {seconds:.2f} 秒です。各ショットのカット時刻、および"
    "(I2VA/FL2VA/L2VA向けの)参照アライメントのタイムスタンプは必ずこの尺の範囲内に収め、"
    "最後のショットは {seconds:.2f} 秒以内に終わること。\n\n"
    "書き換え本文は日本語で出力してください。ただし <d> タグ内の台詞・歌詞、および"
    "画面上のテキストは、ガイド自体のルールどおり原語のまま(翻訳しない)保持すること。"
    "フィールド名自体(integrated_multimodal_description 等の英語名)、[Shot n]・"
    "<Picture n> 等のラベル記法、タイムコード記法(At MM:SS.SS 等)はガイドの英語表記を"
    "そのまま使うこと(これらは構造記法であり翻訳対象ではない)。\n\n"
    "出力は書き換え後のプロンプト本文のみ(該当する場合は先頭の指示行を含む)とすること。"
    "前置き・説明・コードフェンス・「プロンプト:」等のラベルは一切禁止。\n\n"
    "--- スキル手順 ここから ---\n{skill_md}\n--- スキル手順 ここまで ---\n\n"
    "--- リファレンスガイド ({guide_name}) ここから ---\n{guide_text}\n"
    "--- リファレンスガイド ここまで ---"
)


def build_h3_official_system_prompt(task: str, seconds: float, lang: str) -> str:
    if task not in VALID_H3_OFFICIAL_TASKS:
        raise ValueError(f"task は {VALID_H3_OFFICIAL_TASKS} のいずれかです: {task!r}")
    if lang not in VALID_H3_OFFICIAL_LANGS:
        raise ValueError(f"lang は {VALID_H3_OFFICIAL_LANGS} のいずれかです: {lang!r}")

    skill_md = _read_skill_file("SKILL.md")
    if task == "ref2va":
        guide_name = "references/ref-en.txt"
    else:
        guide_name = "references/base-en.txt"
    guide_text = _read_skill_file(Path(guide_name).name)

    wrapper = _H3_OFFICIAL_WRAPPER_JA if lang == "ja" else _H3_OFFICIAL_WRAPPER_EN
    return wrapper.format(
        seconds=float(seconds), skill_md=skill_md, guide_name=guide_name, guide_text=guide_text
    )


def enhance_prompt(
    text: str,
    mode: str,
    seconds: float = 10.0,
    *,
    task: str = "t2va",
    lang: str = "en",
) -> str:
    if mode == "brief":
        return chat_completion(BRIEF_SYSTEM_PROMPT, text)
    if mode == "storyboard":
        sec = int(round(seconds))
        return chat_completion(STORYBOARD_SYSTEM_PROMPT_TEMPLATE.format(seconds=sec), text)
    if mode == "translate":
        return chat_completion(TRANSLATE_SYSTEM_PROMPT, text, temperature=0.2)
    if mode == "h3-official":
        system_prompt = build_h3_official_system_prompt(task, seconds, lang)
        return chat_completion(system_prompt, text, temperature=0.4)
    raise ValueError(f"mode は {VALID_MODES} のいずれかです: {mode!r}")
