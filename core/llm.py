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
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("minimax_h3.llm")

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
    # 人の声を勝手に足さない (2026-08-12 追加)。H3 は音の記述に忠実なので、
    # 「遠くで子供たちの笑い声」のような一文を強化側が足すと実際に声が生成され、
    # ユーザーが台詞を書いていないのに発話入りの動画になる (実機で発生)。
    # なお H3 は人物が映っていると声の指示が無くても声らしい音を出す傾向があり
    # (実測 speech_score 0.298 vs 無人風景 0.125)、このルールだけでは声はゼロに
    # ならない -- 確実に消すには出力側の `mute` を使う。README の該当節参照。
    "**音の記述に人の声(台詞・話し声・笑い声・歓声・呼び込み・歌声など)を入れてよいのは、"
    "ユーザーの入力がそれを明示または明確に含意している場合だけ**。"
    "ユーザーが声について何も言っていない場合は、環境音・自然音・物音・音楽のみで音を構成し、"
    "人の声は一切書かないこと。"
    "出力はプロンプト本文のみ。前置き・説明・引用符・「プロンプト:」等のラベルは一切禁止。"
    "**書き直しは出力前に済ませ、訂正・注釈・言い訳を本文に残さないこと**。"
    "「(100mmに修正)」「※」「←」のようなメタ記述を混ぜてはならない — "
    "誤った値を書いたら、最初から正しい値だけを書いた文を出力すること。"
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

# ガイド本文より後ろに置く「最終指示」。英語版・日本語版で共有する(内容は同一の規則で、
# 表現だけ言語に合わせる)。ここに置く理由と各規則の根拠は _H3_OFFICIAL_WRAPPER_EN の
# 末尾コメントを参照。
_TIMING_RULES_EN = (
    "\n\nFINAL INSTRUCTIONS (highest priority; these override anything above that conflicts):\n"
    "1. SHOT LENGTH: every shot must be long enough to be usable. A shot containing dialogue "
    "needs at least 3.0 seconds; a shot without dialogue needs at least 1.5 seconds. Choose cut "
    "times so that EVERY shot -- including the LAST one, which runs from its cut time to the end "
    "of the video -- satisfies this. Putting a cut at 4.5s in a 5-second video is WRONG because "
    "it leaves the final shot only 0.5 seconds.\n"
    "2. DIALOGUE MUST FIT: spoken Japanese takes roughly 0.25 seconds per character (a 12-character "
    "line takes about 3 seconds). Every line of dialogue must fit inside the shot that contains it. "
    "Never shorten or rewrite the user's dialogue -- instead, give that shot more time.\n"
    "3. NUMBER OF SHOTS: match the shot count to the duration. Around 1-2 shots for 5 seconds, "
    "2-3 for 10 seconds, 3-4 for 15 seconds. Fewer, longer shots are better than many short ones. "
    "A single shot for the whole video is perfectly acceptable when there is one continuous action.\n"
    "4. ONE SHOT, ONE FRAMING: do not change the shot size or the main subject inside a single "
    "shot. If the framing or the subject must change, use a new [Shot n] with its own cut time. "
    "Never write a close-up of one subject and then pan to follow a different subject in the same "
    "shot.\n"
    "5. OFF-SCREEN VOICE: if a speaker is not visible in the shot where they speak, you must use "
    "the exact phrase `says in an off-screen voiceover` and state that the on-screen character's "
    "lips remain closed, exactly as the guide requires.\n"
    "6. SPEAKER IDS: every <d> block must be preceded by a speaker ID such as (S1), and the first "
    "time a speaker appears their voice must be characterised (age, gender, pitch, timbre, pace).\n"
)

_TIMING_RULES_JA = (
    "\n\n【最終指示・時間配分(最優先。上記と矛盾する場合はこちらを優先)】\n"
    "1. ショットの長さ: 各ショットは実用に足る長さが必要。台詞のあるショットは最低3.0秒、"
    "台詞の無いショットは最低1.5秒。**最後のショット(カット時刻から動画の終わりまで)を含めて"
    "全ショット**がこれを満たすようにカット時刻を選ぶこと。5秒の動画で4.5秒にカットを置くのは"
    "誤り(最終ショットが0.5秒しか残らない)。\n"
    "2. 台詞は必ずショット内に収める: 日本語の発話は1文字あたり約0.25秒(12文字で約3秒)。"
    "各台詞は、それが属するショットの尺に収まること。**ユーザーの台詞を短縮・書き換えては"
    "ならない** -- 代わりにそのショットに長い尺を与えること。\n"
    "3. ショット数: 尺に見合った数にすること。5秒なら1〜2、10秒なら2〜3、15秒なら3〜4が目安。"
    "短いショットを多数並べるより、少なく長いショットのほうが良い。連続した1つの動作なら"
    "動画全体を1ショットにしてよい。\n"
    "4. 1ショット1構図: 同一ショット内でショットサイズや主要被写体を変えないこと。画角や被写体を"
    "変える必要があるなら、カット時刻を持つ新しい [Shot n] を立てること。あるショットで"
    "ある被写体のクローズアップを書き、同じショット内で別の被写体を追ってパンする、"
    "という書き方は禁止。\n"
    "5. 画面外の声: 話者がそのショットに映っていない場合は、ガイドの規定どおり "
    "`says in an off-screen voiceover` という定型句を必ず使い、画面内の人物の唇は閉じたままと"
    "明記すること。\n"
    "6. 話者ID: すべての <d> ブロックの直前に (S1) 等の話者IDを置き、初出時にはその声を"
    "特徴づける記述(年齢・性別・声の高さ・音色・話す速さ)を添えること。\n"
)

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
    "--- BEGIN REFERENCE GUIDE ({guide_name}) ---\n{guide_text}\n--- END REFERENCE GUIDE ---\n\n"
    # ガイド本文(15.8KB)より**後ろ**に置く再指示。日本語版で「言語指定は後ろほど強く
    # 効く」ことを実機で確認済みだが、英語版には後方ブロックが無く、尺の制約を含む
    # すべての自前指示が前方でガイドに埋もれていた(2026-08-08 のレビューで発見)。
    # 内容は実測ベースライン(5入力×3回、scripts/probe_h3official_compliance.py)で
    # 実際に出た故障クラスに対応する: 時間配分の違反 40%、うち半分は LLM 側で直せる
    # 配分ミス、および1ショット内での画角の混在(クラスB)。
    + _TIMING_RULES_EN
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
    "--- リファレンスガイド ここまで ---\n\n"
    # 言語指定は必ず「ガイド本文より後ろ」に置くこと。ガイド自体が英語で書かれ英語出力の
    # 実例を並べているため、冒頭にだけ日本語指示を置くと後続のガイドに上書きされて英語で
    # 返ってくる(実機で再現)。後方の指示ほど強く効くため、最終指示として再掲する。
    "【最終指示・最優先】上のリファレンスガイドは英語で書かれ、英語出力の実例を示して"
    "いますが、**本タスクでは書き換え本文を日本語で出力**してください。ただし次のものは"
    "英語/原語のまま変えないこと: (1) フィールド名"
    "(integrated_multimodal_description / overall_soundscape / non_diegetic_music 等)、"
    "(2) [Shot n] や <Picture n> / <Subject n> / <Audio n> などのラベル記法、"
    "(3) タイムコード記法(At MM:SS.SSS)、(4) <d> タグ内の台詞・歌詞と画面上のテキスト"
    "(原語のまま保持)、(5) fully_preserved 等の関係マーカー。"
    "地の文(情景・被写体・動作・カメラ・音の描写)のみ日本語にすること。"
    + _TIMING_RULES_JA
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


# 焦点距離として許可された値 (プロンプトガイドの規定)。H3 のガイドはこの4種のみを想定して
# おり、それ以外を書くと未学習の記述になる。
ALLOWED_FOCAL_MM = (35, 50, 65, 100)


def _sanitize_enhanced(text: str) -> str:
    """強化結果から、LLM が混入させがちな2種の欠陥を機械的に取り除く。

    1. **自己訂正の注釈**: 「85mm(100mmに修正)」のように、誤った値を書いたあと括弧で
       訂正を添える出力が実機で観測された (2026-08-12)。そのまま H3 に渡ると、
       プロンプトにメタ文が混ざる。括弧内が焦点距離を示していればその値を採用し、
       注釈自体は削除する。
    2. **許可外の焦点距離**: 35/50/65/100mm 以外は最も近い許可値に丸める。指示違反が
       残るより、規定内の近い値に寄せたほうが実害が小さい。

    どちらも発火したら logger.info を出す (黙って書き換えない)。システムプロンプト側でも
    禁止しているので、ここは最後の防波堤。
    """
    original = text

    # 1) 「NNmm(MMmmに修正)」形式 -> 「MMmm」
    def _take_correction(m):
        return f"{m.group(2)}mm"

    text = re.sub(r"(\d{2,3})\s*mm\s*[(（][^)）]*?(\d{2,3})\s*mm[^)）]*?[)）]", _take_correction, text)
    # 2) 焦点距離を伴わない注釈括弧 (「(100mmに修正)」が単独で残る等) を落とす
    text = re.sub(r"[(（][^)）]{0,20}(?:に修正|へ修正|修正済|訂正)[^)）]{0,10}[)）]", "", text)

    # 3) 許可外の焦点距離を最も近い許可値へ
    def _snap(m):
        val = int(m.group(1))
        if val in ALLOWED_FOCAL_MM:
            return m.group(0)
        nearest = min(ALLOWED_FOCAL_MM, key=lambda a: abs(a - val))
        return f"{nearest}mm"

    text = re.sub(r"(\d{2,3})\s*mm", _snap, text)

    if text != original:
        logger.info("enhanced prompt sanitized (focal length / self-correction annotation removed)")
    return text


def enhance_prompt(
    text: str,
    mode: str,
    seconds: float = 10.0,
    *,
    task: str = "t2va",
    lang: str = "en",
) -> str:
    if mode == "brief":
        return _sanitize_enhanced(chat_completion(BRIEF_SYSTEM_PROMPT, text))
    if mode == "storyboard":
        sec = int(round(seconds))
        return _sanitize_enhanced(
            chat_completion(STORYBOARD_SYSTEM_PROMPT_TEMPLATE.format(seconds=sec), text)
        )
    if mode == "translate":
        return _sanitize_enhanced(chat_completion(TRANSLATE_SYSTEM_PROMPT, text, temperature=0.2))
    if mode == "h3-official":
        return _sanitize_enhanced(_enhance_h3_official(text, seconds, task=task, lang=lang))
    raise ValueError(f"mode は {VALID_MODES} のいずれかです: {mode!r}")


class InfeasibleInputError(ValueError):
    """入力の台詞が指定尺に収まらない(どのモデルでも解決不能)。400 として返す。"""


# 修復ループの試行回数。1回目で違反があれば、その内容を突きつけて再生成する。
# 実測ベースライン(2026-08-08)では違反 40%、うち半分は尺に収まる配分ミス
# = 具体的に指摘すれば直せる見込みの部類だったため、2回まで再試行する。
H3_OFFICIAL_MAX_REPAIRS = int(os.environ.get("H3_OFFICIAL_MAX_REPAIRS", "2"))

_REPAIR_TEMPLATE = (
    "あなたが直前に出力したプロンプトには次の問題があります:\n\n{report}\n\n"
    "動画の尺は {seconds:.2f} 秒です。上記の問題**だけ**を修正し、それ以外の内容"
    "(情景・被写体・台詞の文言・音の描写)は可能な限り保持してください。"
    "台詞は一字一句変えてはいけません。修正後のプロンプト全文のみを出力してください"
    "(前置き・説明・コードフェンスは禁止)。\n\n"
    "--- 直前の出力 ---\n{previous}"
)


def enhance_prompt_checked(
    text: str,
    seconds: float = 10.0,
    *,
    task: str = "t2va",
    lang: str = "en",
) -> dict:
    """h3-official 生成 + 検証 + 修復ループ。`enhance_prompt` と違い、検証結果も返す。

    返り値: {"result": str, "check": {...}, "attempts": int, "repaired": bool}
    入力の台詞が尺に収まらない場合は `InfeasibleInputError`(LLM に投げる前に判定)。
    """
    return _enhance_h3_official(text, seconds, task=task, lang=lang, want_detail=True)


def _enhance_h3_official(text, seconds, *, task, lang, want_detail=False):
    from core import prompt_check

    # --- LLM に投げる前の実現可能性チェック ---
    # 実測で違反6件中3件がこれ。台詞が尺より長い場合、公式仕様が逐語保持を要求する以上
    # どのモデルでも解決できないので、書き換えを試みずにユーザーへ返す。
    feasible = prompt_check.check_input_feasible(text, seconds)
    if not feasible["feasible"]:
        raise InfeasibleInputError(feasible["advice"])

    system_prompt = build_h3_official_system_prompt(task, seconds, lang)
    result = chat_completion(system_prompt, text, temperature=0.4)
    check = prompt_check.check_prompt(result, seconds, task=task)
    attempts = 1

    # --- 修復ループ: 違反があれば具体的に指摘して再生成 ---
    for _ in range(H3_OFFICIAL_MAX_REPAIRS):
        if check["ok"]:
            break
        logger.info("h3-official: 違反あり、修復を試行 (%d回目): %s",
                    attempts, ",".join(check["violations"]))
        repair_input = _REPAIR_TEMPLATE.format(
            report=check["report"], seconds=float(seconds), previous=result
        )
        try:
            candidate = chat_completion(system_prompt, repair_input, temperature=0.4)
        except Exception:
            logger.exception("h3-official: 修復リクエストが失敗、直前の出力を返す")
            break
        attempts += 1
        cand_check = prompt_check.check_prompt(candidate, seconds, task=task)
        # 悪化させない: 違反が増えるなら採用しない (修復が別の場所を壊す事故を防ぐ)
        if len(cand_check["violations"]) <= len(check["violations"]):
            result, check = candidate, cand_check
        else:
            logger.info("h3-official: 修復案は違反が増えたため破棄 (%d -> %d)",
                        len(check["violations"]), len(cand_check["violations"]))
            break

    if check["violations"]:
        logger.warning("h3-official: 修復後も違反が残存: %s", ",".join(check["violations"]))
    if want_detail:
        return {"result": result, "check": check, "attempts": attempts,
                "repaired": attempts > 1}
    return result
