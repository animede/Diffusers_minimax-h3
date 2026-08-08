"""h3-official 形式のプロンプトを機械的に検証する(クラスA = 決定的に判定できる規則)。

なぜ必要か: 2026-08-08 の実測(`scripts/probe_h3official_compliance.py`、5入力×3回)で、
LLM(gemma4-31B Q4_K_M)の出力は**構造・記法の違反が 0/15** だった一方、**時間配分の
違反が 6/15 (40%)** 出た。しかもその6件は性質が2つに分かれる:

  - **入力が物理的に不可能** (3/6): 台詞の推定発話時間が動画の尺を超えている。公式仕様は
    台詞の逐語保持を要求する(短縮禁止)ので、LLM 側では原理的に解決できない --
    生成前にユーザーへ返すべき情報。`estimate_speech_seconds()` / `check_input_feasible()`
  - **LLM の配分ミス** (3/6): 尺には収まるのに配置が悪い -- 違反内容を LLM へ返して
    再生成すれば直せる見込み(`core/llm.py` の修復ループが本モジュールを使う)

検証は LLM 経由の出力だけでなく**手書きプロンプトにも効く**(生成前チェック)。

規則の根拠は公式スキル `skills_cache/h3-prompt-writing/base-en.txt`:
  F1 必須3フィールドが順序どおり (2.2節)
  F2 [Shot 1] に時刻を付けない (4.2節 "Do not add a timestamp to the first shot")
  F3 [Shot n] (n>=2) のカット時刻が厳密増加 (同上 "strictly increasing cut time")
  F4 全カット時刻が尺の範囲内 (同上 "falls within the video duration")
  F6 <d> タグの開閉対応と言語タグ (4.4節)
  F8 <d> の直前に話者ID (Sn) (4.4節 "Subjects who speak ... use stable IDs")
F5(ショット尺の下限)と F7(台詞がショットに収まる)は公式仕様には無い、本アプリが
足す**実用規則** -- 公式は「カット時刻が尺内」としか言わないため、5秒尺で 4.5秒に
カットして最終ショットが0.5秒、という字義どおりだが使えない出力を許してしまう
(実際に発生した)。

意味的整合性(クラスB。例: 猫のクローズアップが右にパンして歩く二人を追う)は
決定的には判定できないため、近似ルール W1 として**警告**にとどめる(violations には
入れない)。実測では問題の入力3回すべてで発火しており、近似でも拾えることは確認済み。
"""
from __future__ import annotations

import re

# 実用規則のしきい値。公式仕様には無い、本アプリの運用上の下限。
MIN_SHOT_S = 1.5            # 台詞の無いショットの下限
MIN_DIALOGUE_SHOT_S = 3.0   # 台詞のあるショットの下限
# 日本語の発話速度の概算。1文字あたり秒 -- 「うん、散歩にぴったりだね。」(13文字) が
# 約3秒、という体感に合わせた保守的な値。厳密な音声合成ではないので目安として使い、
# エラーではなく「尺を伸ばすか台詞を短く」という助言の根拠にする。
SPEECH_S_PER_CHAR = 0.25

FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF2VA_FIELDS = ("subject_definitions", "summary", "retention_analysis",
                 "detailed_description", "overall_soundscape", "non_diegetic_music")

_SHOTSIZE_WORDS = ("close-up", "medium shot", "medium-wide", "wide shot", "long shot",
                   "full shot", "extreme close-up", "medium close-up",
                   "クローズアップ", "ミディアムショット", "ロングショット", "引きの")

_DIALOGUE_RE = re.compile(r"<d>\s*\[([^\]]*)\]([^<]*)</d>")
_SPEAKER_RE = re.compile(r"\(S\d+(?:\s*,\s*S\d+)*\)")

# 違反コード -> 日本語の説明(ユーザー向け表示と、LLM への修復指示の両方に使う)
MESSAGES = {
    "F1_missing_field": "必須フィールドが欠けています",
    "F1_field_order": "フィールドの順序が公式仕様と異なります",
    "F2_no_shot_label": "[Shot n] のラベルが1つもありません",
    "F2_first_shot_has_time": "[Shot 1] に時刻が付いています(先頭ショットは時刻なしが公式仕様)",
    "F3_missing_cut_time": "[Shot 2] 以降にカット時刻 (At MM:SS.SSS) がありません",
    "F3_not_increasing": "カット時刻が厳密増加していません",
    "F4_cut_out_of_range": "カット時刻が動画の尺の範囲外です",
    "F5_shot_too_short": f"ショットが短すぎます(下限 {MIN_SHOT_S} 秒)",
    "F5_dialogue_shot_too_short": f"台詞のあるショットが短すぎます(下限 {MIN_DIALOGUE_SHOT_S} 秒)",
    "F6_unbalanced_d_tag": "<d> と </d> の数が一致しません",
    "F6_missing_language_tag": "<d> タグに言語タグ([Japanese] 等)がありません",
    "F7_dialogue_longer_than_shot": "台詞がそのショットの尺に収まりません",
    "F8_no_speaker_id": "<d> の直前に話者ID (S1) 等がありません",
}


def estimate_speech_seconds(text: str) -> float:
    """台詞テキストの発話時間を概算する(SPEECH_S_PER_CHAR のコメント参照)。"""
    return len(text.strip()) * SPEECH_S_PER_CHAR


def extract_dialogue(prompt: str) -> list[tuple[str, str]]:
    """`<d>[Lang] text</d>` を (言語, 台詞) の並びで取り出す。"""
    return [(lang.strip(), text.strip()) for lang, text in _DIALOGUE_RE.findall(prompt)]


def check_input_feasible(text: str, seconds: float) -> dict:
    """**LLM に投げる前**の実現可能性チェック: 入力に含まれる台詞(鉤括弧「」または
    <d> タグ)の推定発話時間が尺を超えていないか。

    超えている場合はどんなモデルでも解決できない(公式仕様が逐語保持を要求するため
    短縮も禁止)ので、書き換えを試みる前にユーザーへ返す。実測で 6件中3件がこれだった。

    返り値: {"feasible": bool, "estimated_speech_s": float, "seconds": float,
             "lines": [...], "advice": str|None}
    """
    lines = [t for _, t in extract_dialogue(text)]
    if not lines:
        # 鉤括弧の台詞(ユーザーが公式記法を知らずに書く自然な形)も拾う
        lines = [m.strip() for m in re.findall(r"[「『]([^」』]+)[」』]", text)]
    total = sum(estimate_speech_seconds(t) for t in lines)
    feasible = total <= seconds
    advice = None
    if not feasible:
        need = int(total) + 1
        advice = (
            f"台詞の合計が約 {total:.1f} 秒で、指定の尺 {seconds:.1f} 秒に収まりません。"
            f"尺を {need} 秒以上にするか、台詞を短くしてください"
            f"(公式仕様は台詞の一字一句保持を要求するため、自動短縮はできません)。"
        )
    return {"feasible": feasible, "estimated_speech_s": round(total, 2),
            "seconds": seconds, "lines": lines, "advice": advice}


def _parse_shots(body: str) -> list[dict]:
    shots = []
    for m in re.finditer(r"\[Shot (\d+)\]", body):
        tail = body[m.end():m.end() + 80]
        tm = re.match(r"[ ,]*At (\d+):(\d+)\.(\d+)", tail)
        t = None
        if tm:
            t = int(tm.group(1)) * 60 + int(tm.group(2)) + float("0." + tm.group(3))
        shots.append({"n": int(m.group(1)), "t": t, "start": m.start()})
    for i, s in enumerate(shots):
        s["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else len(body)
        s["text"] = body[s["start"]:s["end"]]
    return shots


def check_prompt(prompt: str, seconds: float, task: str = "t2va") -> dict:
    """h3-official 形式のプロンプトを検証する。

    返り値: {"violations": [code...], "warnings": [str...], "details": {...},
             "ok": bool, "report": str}
    `report` は LLM への修復指示にもユーザー表示にも使える日本語の要約。
    """
    v: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    fields = REF2VA_FIELDS if task == "ref2va" else FIELDS
    positions = {f: prompt.find(f + ":") for f in fields}
    missing = [f for f, p in positions.items() if p < 0]
    if missing:
        v.append("F1_missing_field")
        details["missing_fields"] = missing
    else:
        ordered = [positions[f] for f in fields]
        if ordered != sorted(ordered):
            v.append("F1_field_order")

    # 本文(ショット記述)の範囲: 最初のフィールドから overall_soundscape の手前まで
    body_key = "detailed_description" if task == "ref2va" else FIELDS[0]
    start = positions.get(body_key, -1)
    end = positions.get("overall_soundscape", -1)
    body = prompt[(start if start >= 0 else 0):(end if end > start else len(prompt))]

    shots = _parse_shots(body)
    details["n_shots"] = len(shots)
    if not shots:
        v.append("F2_no_shot_label")
        return _finish(v, warnings, details)

    if shots[0]["t"] is not None:
        v.append("F2_first_shot_has_time")

    times = [s["t"] for s in shots[1:]]
    details["cut_times"] = times
    if any(t is None for t in times):
        v.append("F3_missing_cut_time")
    else:
        if any(b <= a for a, b in zip(times, times[1:])):
            v.append("F3_not_increasing")
        if any(t >= seconds or t <= 0 for t in times):
            v.append("F4_cut_out_of_range")

    bounds = [0.0] + [t for t in times if t is not None] + [seconds]
    durations = [round(bounds[i + 1] - bounds[i], 3) for i in range(min(len(shots), len(bounds) - 1))]
    details["shot_durations"] = durations

    for i, s in enumerate(shots):
        if i >= len(durations):
            break
        dur = durations[i]
        lines = _DIALOGUE_RE.findall(s["text"])
        if lines:
            if dur < MIN_DIALOGUE_SHOT_S:
                v.append("F5_dialogue_shot_too_short")
                details.setdefault("short_shots", []).append({"shot": s["n"], "duration_s": dur})
            speech = sum(estimate_speech_seconds(t) for _, t in lines)
            if speech > dur:
                v.append("F7_dialogue_longer_than_shot")
                details.setdefault("overlong_dialogue", []).append(
                    {"shot": s["n"], "estimated_speech_s": round(speech, 2), "shot_duration_s": dur})
        elif dur < MIN_SHOT_S:
            v.append("F5_shot_too_short")
            details.setdefault("short_shots", []).append({"shot": s["n"], "duration_s": dur})

        low = s["text"].lower()
        if sum(low.count(w) for w in _SHOTSIZE_WORDS) >= 2:
            warnings.append(
                f"[Shot {s['n']}] に複数のショットサイズ指定があります"
                "(1ショット1構図が原則。画角を変えるならカットを分けてください)"
            )

    n_open, n_close = prompt.count("<d>"), prompt.count("</d>")
    n_valid = len(_DIALOGUE_RE.findall(prompt))
    details["n_dialogue"] = n_valid
    if n_open != n_close:
        v.append("F6_unbalanced_d_tag")
    elif n_open != n_valid:
        v.append("F6_missing_language_tag")

    for m in re.finditer(r"<d>", prompt):
        before = prompt[max(0, m.start() - 220):m.start()]
        if not _SPEAKER_RE.search(before):
            v.append("F8_no_speaker_id")
            break

    return _finish(v, warnings, details)


def _finish(violations: list[str], warnings: list[str], details: dict) -> dict:
    violations = sorted(set(violations))
    return {
        "violations": violations,
        "warnings": warnings,
        "details": details,
        "ok": not violations,
        "report": format_report(violations, warnings, details),
    }


def format_report(violations: list[str], warnings: list[str], details: dict) -> str:
    """違反と警告を、LLM への修復指示にもユーザー表示にも使える日本語にまとめる。"""
    if not violations and not warnings:
        return ""
    parts = []
    for code in violations:
        line = f"- {MESSAGES.get(code, code)}"
        if code == "F7_dialogue_longer_than_shot":
            for d in details.get("overlong_dialogue", []):
                line += (f" (Shot {d['shot']}: 台詞は約{d['estimated_speech_s']}秒だが"
                         f"ショットは{d['shot_duration_s']}秒)")
        elif code in ("F5_shot_too_short", "F5_dialogue_shot_too_short"):
            for d in details.get("short_shots", []):
                line += f" (Shot {d['shot']}: {d['duration_s']}秒)"
        elif code == "F1_missing_field":
            line += f" ({', '.join(details.get('missing_fields', []))})"
        elif code == "F4_cut_out_of_range":
            line += f" (カット時刻: {details.get('cut_times')})"
        parts.append(line)
    parts += [f"- {w}" for w in warnings]
    return "\n".join(parts)
