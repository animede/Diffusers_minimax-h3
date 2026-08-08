"""h3-official モードの出力を機械的に検証し、故障クラスの発生率を測る。

背景: 「LLM の性能限界なのか、プロンプトで直るのか」を当てずっぽうで議論しないために、
まず**どのクラスの違反がどの頻度で出るか**のベースラインを取る。温度 0.4 なので同一
入力でも出力はばらつく -- 入力を変え、各入力を複数回試す。

検証するのは**機械的に判定できる規則(クラスA)**のみ。公式スキル
(skills_cache/h3-prompt-writing/base-en.txt) の該当箇所を根拠に:
  F1 必須3フィールドが順序どおり揃っている (2.2節)
  F2 [Shot 1] に時刻が付いていない (4.2節「Do not add a timestamp to the first shot」)
  F3 [Shot n] (n>=2) のカット時刻が厳密増加 (同上「strictly increasing cut time」)
  F4 全カット時刻が尺の範囲内 (同上「falls within the video duration」)
  F5 各ショットに実用的な尺がある (公式には無い、本アプリが足す実用規則。
      台詞ありは MIN_DIALOGUE_SHOT_S、無しは MIN_SHOT_S 以上)
  F6 <d> タグが開閉対応し、言語タグ [Xxx] を持つ (4.4節)
  F7 台詞が属するショットの尺に収まる (発話速度 SPEECH_S_PER_CHAR で概算)
  F8 <d> の直前に話者ID (Sn) がある (4.4節「Subjects who speak ... use stable IDs」)
  F9 文脈溢れが起きていない (system+入力+出力 が n_ctx 以内)

意味的整合性(クラスB。例: 猫のクローズアップが右にパンして歩く二人を追う)は
ここでは判定しない -- 近似ルール F10 として「同一ショット内にショットサイズ語が
複数出る」だけを警告として数える。

実行 (GPU不要、LLM だけ使う):
  venv/bin/python scripts/probe_h3official_compliance.py
  H3_PROBE_REPEATS=3 H3_PROBE_TAG=baseline venv/bin/python scripts/probe_h3official_compliance.py
"""
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "outputs" / "ab_h3official_compliance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 実用規則のしきい値 (公式仕様には無い、本アプリの運用上の下限)
MIN_SHOT_S = 1.5            # 台詞の無いショットの下限
MIN_DIALOGUE_SHOT_S = 3.0   # 台詞のあるショットの下限
SPEECH_S_PER_CHAR = 0.25    # 日本語の発話速度の概算 (1文字あたり秒)

# (label, seconds, 入力テキスト)。台詞の有無・カット数・尺・曖昧さを散らす。
CASES = [
    ("d1_short", 5.0,
     "制服の少女が坂道を歩きながら「今日はいい天気だね」と言う。明るい住宅街。"),
    ("d2_multi", 10.0,
     "少女と親友が坂道を歩いて会話する。少女「今日は本当にいい天気だね」親友「うん、"
     "散歩にぴったり」。途中で日向ぼっこする猫を挟み、最後に二人が振り返って笑う。"),
    ("d3_long_line", 9.0,
     "喫茶店で男性が窓の外を見ながら独白する。「あの日の帰り道、彼女が最後に言った言葉を、"
     "僕は今でもはっきりと覚えているんだ」。雨音。"),
    ("n1_scenic", 5.0,
     "夜明けの雪原を狐が歩く。鳥の声と風の音。台詞なし。"),
    ("n2_vague", 8.0,
     "海辺の夏の思い出。"),
]

REPEATS = int(os.environ.get("H3_PROBE_REPEATS", "3"))
TAG = os.environ.get("H3_PROBE_TAG", "baseline")

FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
SHOTSIZE_WORDS = ("close-up", "medium shot", "medium-wide", "wide shot", "long shot",
                  "full shot", "extreme close-up", "medium close-up")


def _tokenize_count(llm_url: str, text: str) -> int | None:
    try:
        req = urllib.request.Request(
            f"{llm_url}/tokenize", data=json.dumps({"content": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])
    except Exception:
        return None


def _parse_shots(body: str) -> list[dict]:
    """[Shot n] とカット時刻を抽出する。時刻は `At MM:SS.SSS` 記法。"""
    shots = []
    for m in re.finditer(r"\[Shot (\d+)\]", body):
        n = int(m.group(1))
        tail = body[m.end():m.end() + 80]
        tm = re.match(r"[ ,]*At (\d+):(\d+)\.(\d+)", tail)
        t = None
        if tm:
            t = int(tm.group(1)) * 60 + int(tm.group(2)) + float("0." + tm.group(3))
        shots.append({"n": n, "t": t, "start": m.start()})
    for i, s in enumerate(shots):
        s["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else len(body)
        s["text"] = body[s["start"]:s["end"]]
    return shots


def check(result: str, seconds: float) -> dict:
    """クラスA の違反を列挙する。返り値の "violations" は違反コードのリスト。"""
    v = []
    notes = {}

    # --- F1: 必須3フィールド ---
    positions = {f: result.find(f + ":") for f in FIELDS}
    missing = [f for f, p in positions.items() if p < 0]
    if missing:
        v.append("F1_missing_field")
        notes["missing_fields"] = missing
    elif not (positions[FIELDS[0]] < positions[FIELDS[1]] < positions[FIELDS[2]]):
        v.append("F1_field_order")

    body_start = positions[FIELDS[0]] if positions[FIELDS[0]] >= 0 else 0
    body_end = positions[FIELDS[1]] if positions[FIELDS[1]] >= 0 else len(result)
    body = result[body_start:body_end]

    shots = _parse_shots(body)
    notes["n_shots"] = len(shots)
    if not shots:
        v.append("F2_no_shot_label")
        return {"violations": v, "notes": notes}

    # --- F2: 先頭ショットに時刻を付けない ---
    if shots[0]["t"] is not None:
        v.append("F2_first_shot_has_time")

    # --- F3/F4: カット時刻の厳密増加と尺内 ---
    times = [s["t"] for s in shots[1:]]
    notes["cut_times"] = times
    if any(t is None for t in times):
        v.append("F3_missing_cut_time")
    else:
        if any(b <= a for a, b in zip(times, times[1:])):
            v.append("F3_not_increasing")
        if any(t >= seconds or t <= 0 for t in times):
            v.append("F4_cut_out_of_range")

    # --- F5/F7: ショット尺と台詞の収まり ---
    bounds = [0.0] + [t for t in times if t is not None] + [seconds]
    durations = []
    for i, s in enumerate(shots):
        if i + 1 < len(bounds):
            durations.append(round(bounds[i + 1] - bounds[i], 3))
    notes["shot_durations"] = durations

    dlg_re = re.compile(r"<d>\s*\[([^\]]*)\]([^<]*)</d>")
    for i, s in enumerate(shots):
        dur = durations[i] if i < len(durations) else None
        lines = dlg_re.findall(s["text"])
        has_dlg = bool(lines)
        if dur is not None:
            if has_dlg and dur < MIN_DIALOGUE_SHOT_S:
                v.append("F5_dialogue_shot_too_short")
                notes.setdefault("short_shots", []).append({"shot": s["n"], "dur": dur, "dialogue": True})
            elif not has_dlg and dur < MIN_SHOT_S:
                v.append("F5_shot_too_short")
                notes.setdefault("short_shots", []).append({"shot": s["n"], "dur": dur, "dialogue": False})
            speech = sum(len(t.strip()) * SPEECH_S_PER_CHAR for _, t in lines)
            if speech > dur:
                v.append("F7_dialogue_longer_than_shot")
                notes.setdefault("overlong_dialogue", []).append(
                    {"shot": s["n"], "est_speech_s": round(speech, 2), "shot_dur": dur})

    # --- F6: <d> の開閉と言語タグ ---
    n_open, n_close = result.count("<d>"), result.count("</d>")
    if n_open != n_close:
        v.append("F6_unbalanced_d_tag")
    n_valid = len(dlg_re.findall(result))
    notes["n_dialogue"] = n_valid
    if n_open != n_valid:
        v.append("F6_missing_language_tag")

    # --- F8: <d> の直前に話者ID ---
    for m in re.finditer(r"<d>", result):
        before = result[max(0, m.start() - 220):m.start()]
        if not re.search(r"\(S\d+(?:\s*,\s*S\d+)*\)", before):
            v.append("F8_no_speaker_id")
            break

    # --- F10 (参考、クラスB近似): 同一ショット内に複数のショットサイズ語 ---
    for s in shots:
        low = s["text"].lower()
        if sum(low.count(w) for w in SHOTSIZE_WORDS) >= 2:
            notes.setdefault("multi_framing_shots", []).append(s["n"])

    return {"violations": sorted(set(v)), "notes": notes}


def main():
    from core.llm import build_h3_official_system_prompt, enhance_prompt, get_llm_url

    llm_url = get_llm_url()
    logging.info("LLM: %s / repeats=%d / tag=%s", llm_url, REPEATS, TAG)

    props = json.load(urllib.request.urlopen(f"{llm_url}/props", timeout=30))
    n_ctx = props["default_generation_settings"]["n_ctx"]
    logging.info("n_ctx = %d", n_ctx)

    records = []
    for label, seconds, text in CASES:
        sp_tokens = _tokenize_count(llm_url, build_h3_official_system_prompt("t2va", seconds, "en"))
        in_tokens = _tokenize_count(llm_url, text)
        for rep in range(REPEATS):
            t0 = time.time()
            rec = {"label": label, "rep": rep, "seconds": seconds, "tag": TAG,
                   "system_tokens": sp_tokens, "input_tokens": in_tokens}
            try:
                result = enhance_prompt(text, "h3-official", seconds=seconds, task="t2va", lang="en")
                out_tokens = _tokenize_count(llm_url, result)
                rec["output_tokens"] = out_tokens
                total = (sp_tokens or 0) + (in_tokens or 0) + (out_tokens or 0)
                rec["total_tokens"] = total
                rec["ctx_headroom"] = n_ctx - total
                res = check(result, seconds)
                rec.update(res)
                if rec["ctx_headroom"] < 0:
                    rec["violations"] = sorted(set(rec["violations"] + ["F9_ctx_overflow"]))
                rec["result"] = result
            except Exception as e:
                rec["exception"] = f"{type(e).__name__}: {e}"
                # InfeasibleInputError は**正しい挙動**(台詞が尺に収まらない入力を
                # LLM に投げる前に弾き、ユーザーへ助言を返す)。違反として数えると
                # 改善が退行に見えてしまうので、専用の分類にする。
                if type(e).__name__ == "InfeasibleInputError":
                    rec["violations"] = []
                    rec["infeasible_input"] = True
                else:
                    rec["violations"] = ["EXCEPTION"]
            rec["elapsed_s"] = round(time.time() - t0, 2)
            records.append(rec)
            logging.info("%s rep%d (%.1fs): %s", label, rep, rec["elapsed_s"],
                         ",".join(rec.get("violations", [])) or "CLEAN")

    out = OUT_DIR / f"compliance_{TAG}.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    # --- 集計 ---
    total = len(records)
    infeasible = sum(1 for r in records if r.get("infeasible_input"))
    clean = sum(1 for r in records if not r.get("violations") and not r.get("infeasible_input"))
    unresolved = sum(1 for r in records if r.get("violations"))
    counts: dict[str, int] = {}
    for r in records:
        for code in r.get("violations", []):
            counts[code] = counts.get(code, 0) + 1
    logging.info("=== summary (tag=%s, n=%d) ===", TAG, total)
    logging.info("clean:            %d/%d (%.0f%%)", clean, total, 100 * clean / total)
    logging.info("入力不可を検出:    %d/%d (正しい挙動: 生成前にユーザーへ助言)", infeasible, total)
    logging.info("違反が残存:        %d/%d", unresolved, total)
    for code, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        logging.info("  %-32s %d/%d (%.0f%%)", code, c, total, 100 * c / total)
    mh = min((r.get("ctx_headroom", 10**9) for r in records), default=None)
    logging.info("min ctx headroom: %s tokens", mh)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
