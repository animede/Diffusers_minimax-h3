#!/usr/bin/env python
"""低 VRAM・低 RAM の PC 向けの準備(初回のみ)。ホスト RAM 16GB・VRAM 8GB クラスでも作れる。

`H3_LOWVRAM=group`(32GB 以下の級)で重みを量子化キャッシュのファイルから直接流す構成(ホスト RAM 48GB 未満で
自動選択される `H3_GROUP_STREAM_HOST=file`)は、`models/prequant/transformer_int8`(約 32GB)が **先に存在する**
ことが前提になっている。無いと bf16(66GB)から量子化する従来経路に落ち、ホスト RAM 40GB のガードで止まる。
このスクリプトはそのキャッシュを、bf16 のシャード(14 個、各 4.7GB)を **1 つずつ落として量子化して消す** 方式で作る
(RAM は数 GB、ディスクのピークはシャード 1 つ分 + 出力)。手順:

  1. 本家 MiniMaxAI/MiniMax-H3 の小物(vae / audio_vae / scheduler / tokenizer / processor / 各 config)を HF キャッシュへ。
     32B の text_encoder(66.7GB)は落とさない(低 VRAM 級は投影 TE を使うため不要)。
  2. transformer の int8 量子化キャッシュ(models/prequant/transformer_int8)をシャード単位で構築。
     `--ref` を付けると Ref2VA / Ref2I 用の transformer_ref(transformer とは別の重み、66GB)も同じ方式で
     models/prequant/transformer_ref_int8 に作る(+32GB)。低 RAM の PC は、これが無いと参照モードを使えない。
     torchao の Int8WeightOnlyConfig(version=2) をサーバと同じ条件で掛け、サーバの save_pretrained と同じ
     safetensors 形式(torchao の flatten)で書くので、出来上がりはサーバが自分で作るものと同じ。
  3. video VAE デコーダの fp16 キャッシュ(models/prequant/vae_decoder_fp16)。
  4. 投影 TE(Qwen/Qwen3-VL-4B-Instruct + NicoLab28/ClipProj-MiniMax-H3)を先に取得(NF4 キャッシュは初回起動時に
     サーバが作る。VRAM 8GB・RAM 16GB で可)。

使い方(Linux は venv/bin/python、Windows は .\\venv\\Scripts\\python.exe):
  venv/bin/python scripts/prepare_lowvram.py                 # 1〜4 すべて
  venv/bin/python scripts/prepare_lowvram.py --keep-source   # bf16 シャードを消さずに残す(再構築が速い。+66GB)
  venv/bin/python scripts/prepare_lowvram.py --device cuda   # 量子化を GPU で(既定は cpu。結果は同じ)
  venv/bin/python scripts/prepare_lowvram.py --skip-vae --skip-te   # transformer だけ
  venv/bin/python scripts/prepare_lowvram.py --ref           # Ref2VA / Ref2I 用の transformer_ref(別の 66GB)のキャッシュも作る(+32GB)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.hubenv  # noqa: E402,F401  Windows: HF キャッシュの symlink を使わない(WinError 1314 回避)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("minimax_h3.prepare_lowvram")

MODEL_ID = "MiniMaxAI/MiniMax-H3"
TE_PROJ_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TE_PROJ_REPO = "NicoLab28/ClipProj-MiniMax-H3"
TE_PROJ_FILE = os.environ.get("H3_TE_PROJ_FILE", "mmh3-4b-ClipProj.safetensors").strip()
ROOT = Path(__file__).resolve().parent.parent
PREQUANT_DIR = Path(os.environ.get("H3_TE_PREQUANT_DIR", str(ROOT / "models" / "prequant")))
# core/runner.py の H3_INT8_MODULES_TO_NOT_CONVERT(定義時点の pristine 値)と同じ。meta.json の照合キーになる
MODULES_TO_NOT_CONVERT = ["audio_proj_in", "audio_proj_out", "context_embedder", "norm_out", "proj_in", "proj_out",
                          "time_embedder", "time_proj", "token_refiner"]
# サーバ(diffusers の TorchAoConfig)が config.json に書く quantization_config と同一
QUANTIZATION_CONFIG = {
    "modules_to_not_convert": ["proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                               "token_refiner", "norm_out", "proj_out", "audio_proj_out", "proj_in", "audio_proj_in",
                               "time_embedder", "proj_out", "audio_proj_out", "rope"],
    "quant_method": "torchao",
    "quant_type": {"default": {"_data": {"granularity": {"_data": {"dim": -1}, "_type": "PerRow", "_version": 1},
                                         "group_size": None, "set_inductor_config": False},
                               "_type": "Int8WeightOnlyConfig", "_version": 2}},
}
SMALL_PATTERNS = ["*.json", "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*", "tokenizer/*", "processor/*",
                  "text_encoder/*.json", "transformer/config.json", "transformer/*.index.json"]


def _release(path: Path) -> float:
    """HF キャッシュのシャード(symlink + blob)を消して空けた GB を返す。"""
    real = Path(os.path.realpath(path))
    size = real.stat().st_size if real.exists() else 0
    for q in (path, real):
        try:
            q.unlink()
        except FileNotFoundError:
            pass
    return size / 1e9


def fetch_small() -> Path:
    from huggingface_hub import snapshot_download

    logger.info("小物を取得: %s %s", MODEL_ID, SMALL_PATTERNS)
    return Path(snapshot_download(MODEL_ID, allow_patterns=SMALL_PATTERNS))


# ---------------------------------------------------------------------------
# transformer int8
# ---------------------------------------------------------------------------
def quantize_linear_weight(w, device: str):
    """サーバの TorchAoConfig(Int8WeightOnlyConfig(version=2, set_inductor_config=False))と同じ量子化を
    1 つの Linear 重みに掛けて torchao の Int8Tensor(CPU)を返す。"""
    import torch
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    out_f, in_f = w.shape
    lin = torch.nn.Linear(in_f, out_f, bias=False, device="meta", dtype=torch.bfloat16)
    lin.weight = torch.nn.Parameter(w.to(device=device, dtype=torch.bfloat16), requires_grad=False)
    quantize_(lin, Int8WeightOnlyConfig(version=2, set_inductor_config=False))
    q = lin.weight
    if device != "cpu":
        q = q.to("cpu")
    return q


def finalize_metadata(cache_dir: Path) -> dict:
    """全シャードのヘッダを「全テンソルぶんのメタデータ + tensor_names」に揃える(diffusers の save_pretrained と同じ)。
    各シャードは自分のテンソルだけを持ち、メタデータは全体で共通。統合したメタデータを返す。"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    files = sorted(cache_dir.glob("*.safetensors"))
    full, names, seen = {}, [], set()
    for fp in files:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            m = dict(f.metadata() or {})
        for name in json.loads(m.pop("tensor_names", "[]")):
            if name not in seen:            # 統一済みのシャードは全名を持つので重複を除く(冪等にする)
                seen.add(name)
                names.append(name)
        m.pop("format", None)
        full.update(m)
    full["tensor_names"] = json.dumps(names)
    full["format"] = "pt"   # safetensors.torch.save_file(..., metadata) が付ける既定と同じ
    rewritten = 0
    for fp in files:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            if dict(f.metadata() or {}) == full:
                continue
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        tmp = fp.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=full)
        os.replace(tmp, fp)
        del tensors
        rewritten += 1
    if rewritten:
        logger.info("メタデータを %d シャードで統一(%d テンソル、%d ファイルを書き直し)", len(files), len(names), rewritten)
    return full


def build_transformer_int8(out_dir: Path, device: str, only_shards: list[int] | None, keep_source: bool,
                           subfolder: str = "transformer") -> None:
    """`subfolder` は "transformer"(T2VA / FL2VA)か "transformer_ref"(Ref2VA / Ref2I、別の重み 66GB)。"""
    import torch
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict

    if (out_dir / "config.json").exists() and (out_dir / "meta.json").exists() and only_shards is None:
        # 既にある: 旧版(シャードごとの部分メタデータ)で作ったものなら、ヘッダだけ統合形式に直す(冪等、約 1 分)
        full = finalize_metadata(out_dir)
        idx_path = out_dir / "diffusion_pytorch_model.safetensors.index.json"
        idx = json.loads(idx_path.read_text())
        if idx.get("metadata", {}).get("tensor_names") != full["tensor_names"]:
            idx["metadata"] = {**full, "total_size": idx["metadata"]["total_size"]}
            idx_path.write_text(json.dumps(idx, indent=2))
        logger.info("%s int8 キャッシュは既にあるのでスキップ(メタデータは確認済み): %s", subfolder, out_dir)
        return
    cfg_path = Path(hf_hub_download(MODEL_ID, f"{subfolder}/config.json"))
    index = json.loads(Path(hf_hub_download(MODEL_ID, f"{subfolder}/diffusion_pytorch_model.safetensors.index.json")).read_text())
    shards = sorted(set(index["weight_map"].values()))
    if only_shards:
        shards = [s for i, s in enumerate(shards, 1) if i in only_shards]
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    weight_map, total = {}, 0
    t_all = time.time()
    for n, shard in enumerate(shards, 1):
        t0 = time.time()
        src = Path(hf_hub_download(MODEL_ID, f"{subfolder}/{shard}"))
        tensors, n_q = {}, 0
        with safe_open(str(src), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key.startswith("transformer_blocks.") and t.ndim == 2:
                    tensors[key] = quantize_linear_weight(t, device)
                    n_q += 1
                else:
                    tensors[key] = t
        flat, meta = flatten_tensor_state_dict(tensors)
        flat = {k: v.contiguous() for k, v in flat.items()}
        save_file(flat, str(tmp / shard), metadata=meta)
        for k, v in flat.items():
            weight_map[k] = shard
            total += v.numel() * v.element_size()
        del tensors, flat
        freed = 0.0 if keep_source else _release(src)
        logger.info("[%d/%d] %s/%s: %d 層を int8 化、%.1fs%s", n, len(shards), subfolder, shard, n_q, time.time() - t0,
                    f"(原本 {freed:.1f}GB を削除)" if freed else "")
    # diffusers の save_pretrained は、全テンソルぶんの torchao メタデータ(+ tensor_names)を**全シャードの
    # ヘッダに同じ内容で**書く。ロード側はその前提で 1 つのメタデータから全体を復元するので、シャードごとの
    # 部分メタデータだと他シャードの重みが初期化されず「Cannot copy out of meta tensor」で落ちる(実測)。
    # 各シャードを書き終えてから、全体のメタデータで全シャードのヘッダを書き直す。
    full_meta = finalize_metadata(tmp)
    (tmp / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {**full_meta, "total_size": total}, "weight_map": weight_map}, indent=2))
    import diffusers
    import torchao

    cfg = json.loads(cfg_path.read_text())
    cfg["_diffusers_version"] = diffusers.__version__
    cfg["_name_or_path"] = MODEL_ID
    cfg["quantization_config"] = QUANTIZATION_CONFIG
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    # サーバは transformer_ref 側のキャッシュも transformer/config.json のスナップショットで識別する
    # (core/runner.py の _transformer_prequant_metadata)。同じ値を書く
    cached = try_to_load_from_cache(MODEL_ID, "transformer/config.json")
    (tmp / "meta.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "source_snapshot": str(cached) if cached else None,
        "torchao_version": torchao.__version__,
        "torch_version": torch.__version__,
        "quant_config": "Int8WeightOnlyConfig(version=2)",
        "modules_to_not_convert": sorted(MODULES_TO_NOT_CONVERT),
    }, indent=2))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("%s int8 キャッシュ完成: %s(%d シャード、%.1fGB、%.0fs)", subfolder, out_dir, len(shards), total / 1e9,
                time.time() - t_all)


# ---------------------------------------------------------------------------
# video VAE decoder fp16
# ---------------------------------------------------------------------------
def build_vae_fp16(out_dir: Path) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    from core import vaecache

    path = out_dir / vaecache.CACHE_FILE
    if path.exists() and (out_dir / "meta.json").exists():
        logger.info("VAE fp16 キャッシュは既にあるのでスキップ: %s", out_dir)
        return
    vae_dir = Path(snapshot_download(MODEL_ID, allow_patterns=["vae/*"])) / "vae"
    files = sorted(vae_dir.glob("*.safetensors"))
    prefix = "decoder.transformer_blocks."
    tensors, blocks = {}, set()
    for fp in files:
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    tensors[key] = f.get_tensor(key).to(torch.float16).contiguous()
                    blocks.add(int(key[len(prefix):].split(".")[0]))
    # numel はパラメータだけを数える(サーバの vaecache._meta と同じ)。メタデバイスで組み立てて名前を得る
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(AutoencoderKLMiniMaxH3.load_config(str(vae_dir)))
    param_names = {prefix + f"{i}." + n for i, b in enumerate(vae.decoder.transformer_blocks) for n, _ in b.named_parameters()}
    numel = sum(t.numel() for k, t in tensors.items() if k in param_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
    (out_dir / "meta.json").write_text(json.dumps({"dtype": "float16", "blocks": len(blocks), "numel": int(numel),
                                                   "name_or_path": MODEL_ID}, indent=1))
    logger.info("VAE fp16 キャッシュ完成: %s(%d ブロック、%d テンソル)", path, len(blocks), len(tensors))


def prefetch_te_proj() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    logger.info("投影 TE を取得: %s(約 8.8GB)+ %s/%s", TE_PROJ_MODEL, TE_PROJ_REPO, TE_PROJ_FILE)
    snapshot_download(TE_PROJ_MODEL)
    hf_hub_download(TE_PROJ_REPO, TE_PROJ_FILE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="量子化を行うデバイス(cpu / cuda。結果は同じ)")
    ap.add_argument("--keep-source", action="store_true", help="bf16 シャードを削除せず HF キャッシュに残す")
    ap.add_argument("--only-shards", default="", help="デバッグ用: 1,2 のように番号指定したシャードだけ処理")
    ap.add_argument("--out", default="", help="transformer int8 キャッシュの出力先(既定 models/prequant/transformer_int8)")
    ap.add_argument("--ref", action="store_true",
                    help="Ref2VA / Ref2I 用の transformer_ref(別の重み 66GB)の int8 キャッシュ(models/prequant/transformer_ref_int8、"
                         "約 32GB)も作る。低 RAM の PC で参照モードを使うなら必要")
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--skip-te", action="store_true")
    ap.add_argument("--finalize-only", default="", help="内部用: 既に書いたキャッシュのメタデータ統一だけを行う")
    args = ap.parse_args()
    if args.finalize_only:
        d = Path(args.finalize_only)
        full = finalize_metadata(d)
        idx_path = d / "diffusion_pytorch_model.safetensors.index.json"
        idx = json.loads(idx_path.read_text())
        idx["metadata"] = {**full, "total_size": idx["metadata"]["total_size"]}
        idx_path.write_text(json.dumps(idx, indent=2))
        return
    only = [int(x) for x in args.only_shards.split(",") if x.strip()] or None
    out = Path(args.out) if args.out else PREQUANT_DIR / "transformer_int8"
    if not args.skip_small:
        fetch_small()
    if not args.skip_transformer:
        build_transformer_int8(out, args.device, only, args.keep_source)
    if args.ref:
        build_transformer_int8(PREQUANT_DIR / "transformer_ref_int8", args.device, only, args.keep_source,
                               subfolder="transformer_ref")
    if not args.skip_vae:
        build_vae_fp16(PREQUANT_DIR / "vae_decoder_fp16")
    if not args.skip_te:
        prefetch_te_proj()
    logger.info("完了。起動: Linux は venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611、Windows は .\\run.bat")


if __name__ == "__main__":
    main()
