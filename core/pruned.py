"""AdaLN-pruned MiniMax-H3 transformer_ref の読み込みヘルパー(H3_PRUNED=1)。

multimodalart/MiniMax-H3-Pruned(Apache-2.0)は、公式 H3 の AdaLN 入力射影
(各ブロックの `Linear(2688->96768)`、計 13.03B パラメータ = checkpoint の 39.3%)を
到達可能ランク(8次元)へ構造リファクタしたもの。近似誤差 ~1.5e-5(bf16 丸め
1step の 1/250)でほぼロスレス。35B -> 20.1B、bf16 66.3GB -> 40.2GB。

量子化方式は起動時環境変数 `H3_PRUNED_QUANT`(runner.py の config ブロック、値の一覧は
`PRUNED_QUANT_SPECS`)で選ぶ。既定は `int8wo`(torchao Int8WeightOnlyConfig(version=2,
PerRow)、ConvRot なし。非 pruned 本番 int8 と同じ重みレシピ)。2026-10-01 の実測で
4.60 s/step・peak 28GB・品質は int8dyn-convrot と同等(旧既定 int8dyn-convrot は 7.60
s/step)。速度差の主因は、torchao 0.17 の int8 動的量子化 eager 経路(未融合の per-token
量子化 + int32 逆量子化)であり ConvRot の回転コスト(step の ~2%)ではない(GEMM
マイクロベンチ)。`int8dyn-convrot` は remote code の `quantize_8bit()`(ConvRot =
Hadamard 条件付け + Int8DynamicActivationInt8WeightConfig、「bf16 に最も近いと実測された
構成」とモデルカードに明記)で、常駐 ~19.6GiB(フェーズA probe 実測)。選択肢として残す。

既存機構との互換性(フェーズA probe + ソース照合で確認済み):
- turbo LoRA(_TurboLoRALinear): 互換。lightx2v ref2v turbo は AdaLN キーを
  一切持たず(312 = blocks 300 + token refiner 12、全て attn/ff)、ラッパーは
  `self.base(x)` 経由なので ConvRot の入力回転も保持される。量子化/mark は
  turbo LoRA の wrap より前に完了していること(`convrot_layers()` は nn.Linear
  判定のため、wrap 後の `….base` は拾えない)。
- FBC / attention backend: ブロッククラスは公式実装を継承しており互換。
- H3_ADALN_PRECOMP: **非互換**(pruned は AdaLN 構造自体が別物で、精計算の
  対象が存在しない)。runner 側で pruned 時はスキップする。
- HyperFlow(H3_HYPERFLOW): **非互換**(pruned は time_proj/time_embedder MLP を
  補間テーブルに置換しており、TwoTimeEmbedder のラップ対象が無い)。runner 側で
  併用を起動時に拒否する。

キャッシュ戦略は H3_TRANSFORMER_PREQUANT と同じ(量子化済みを保存、以後は読むだけ)。
方式ごとにディレクトリ名・state ファイル名・meta.json の `quant_config` を分けており
(int8dyn-convrot は旧既定時代の名前・内容のまま)、別方式のキャッシュを取り違えて読むことはない。
ConvRot 系の方式では「オンライン入力回転」がモジュールの __class__ 差し替えで実現
されており直列化されないため、キャッシュからの再ロード後に `mark_convrot()` で
**重みを再回転せずクラスだけ**を付け直す(enable_convrot を再実行すると Hadamard が
対合のため重みが元に戻ってしまい不正になる -- remote code の enable_convrot docstring
参照)。ConvRot 無しの方式では mark_convrot を**呼んではいけない**(回転していない重みに
付けると入力だけ回転して無言で誤った出力になる)。どちらの誤りも無言で壊れるため、
呼ぶ/呼ばないの判断は `PrunedQuantSpec.convrot` だけから機械的に決める。
bf16 はキャッシュしない(スナップショットそのものがキャッシュ)。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger("minimax_h3.pruned")

_MODULE_CACHE: dict[str, object] = {}


# ---- H3_PRUNED_QUANT の方式定義 ------------------------------------------------------

def _cfg_int8dyn():
    # remote code `quantize_8bit(config=None)` が内部で作るものと同一(dataclass 既定の
    # version=1、PlainLayout/AffineQuantizedTensor 経路)。量子化結果(重み)を変えないため
    # 付ける引数は set_inductor_config=False だけ(下記)。
    #
    # set_inductor_config: torchao の既定 True だと quantize_() が
    # `recommended_inductor_config_setter()` を呼び、`torch.set_float32_matmul_precision("high")`
    # (= fp32 matmul の TF32 化)と inductor 設定を**プロセス全体**に立てる。fresh 量子化を
    # 一度でもしたプロセスだけ以後の fp32 計算(audio_vae 等)の数値が変わり、キャッシュ
    # 読み込みだけのプロセスと出力が一致しなくなる。この値は量子化済みテンソルには
    # 影響しない(フラグの副作用だけ)ので False にする。以下の factory も同様。
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    return Int8DynamicActivationInt8WeightConfig(set_inductor_config=False)


def _cfg_int8wo():
    # 非 pruned 本番 int8(runner の `Int8WeightOnlyConfig(version=2)`)と同一レシピ
    # (torchao 0.17.0 では version=2 の granularity 既定が PerRow(dim=-1)、実機確認済み。
    # 明示しているのは meta.json の文字列と読み合わせやすくするため)。
    from torchao.quantization import Int8WeightOnlyConfig, PerRow

    return Int8WeightOnlyConfig(version=2, granularity=PerRow(), set_inductor_config=False)


def _cfg_fp8dyn():
    # per-row fp8 活性化 + fp8 重み -> torch._scaled_mm(out_dtype=bf16)。Float8Tensor の
    # 量子化は CUDA 必須(CPU では AssertionError)なので、層単位で GPU へ上げる流儀が前提。
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), set_inductor_config=False)


def _cfg_int4wo():
    # W4(tinygemm、活性は bf16 のまま = w4a16)。Kijai/MiniMax-H3-experimental の
    # w4a8_mixed(comfy_quant コードブック形式、ComfyUI カーネル専用で直接ロード不可)を
    # 受けた「4bit 重みで品質が保てるか」の自前プローブ用(2026-10-07)。活性 a16 は
    # a8 より高精度なので、これで品質が崩れるなら w4a8 はなお悪い、という安全側の判定。
    # group_size=128 は tinygemm の標準(Kijai は group16 コードブックでより細かい)。
    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows import Int4PackingFormat

    # 既定の PLAIN パッキングは外部カーネル mslk を要求する(未導入、実機で
    # 「Requires mslk >= 1.0.0」)。torch 内蔵 tinygemm(_weight_int4pack_mm)を使う
    # TILE_PACKED_TO_4D に切り替える(bf16 活性前提 = 本構成と一致)。
    return Int4WeightOnlyConfig(group_size=128, set_inductor_config=False,
                                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)


@dataclass(frozen=True)
class PrunedQuantSpec:
    """`H3_PRUNED_QUANT` の 1 値ぶんのレシピ。

    convrot: remote code の ConvRot(重みへの Hadamard fold + オンライン入力回転)を
        掛けるか。True なら `quantize_8bit()` 経由で量子化し、キャッシュ再ロード時に
        `mark_convrot()` を呼ぶ。False なら `_quantize_plain()` で回転なしに量子化し、
        mark_convrot は呼ばない。
    cache_name / state_filename: 量子化済みキャッシュのディレクトリ名と state_dict
        ファイル名(None = キャッシュしない)。
    meta_quant_config: meta.json の `quant_config`(キャッシュ無効化キー)。
    config_factory: torchao config を作る関数(None = 量子化しない)。torchao の import を
        呼び出し時まで遅らせるため関数にしている。
    """

    name: str
    convrot: bool
    cache_name: str | None
    state_filename: str | None
    meta_quant_config: str
    config_factory: Callable[[], object] | None
    # comfy-kitchen 経路(torchao を使わない)。True なら量子化は
    # `_quantize_kitchen()`(ck.quantize_w4a8_int8_weight + CKW4A8Linear 差し替え)、
    # キャッシュ再ロードは `_install_kitchen_shells()` を経由する。convrot は必ず False
    # (kitchen の w4a8 カーネルは convrot_groupsize=256 の回転を内部に持つ自己完結設計で、
    # remote code の mark_convrot と併用してはいけない)。
    kitchen: bool = False

    @property
    def quantized(self) -> bool:
        return self.config_factory is not None or self.kitchen

    @property
    def cacheable(self) -> bool:
        return self.cache_name is not None

    def make_config(self):
        return None if self.config_factory is None else self.config_factory()

    def check_available(self) -> None:
        """起動時に torchao 側へ config が存在するか確認する(無ければ明確な RuntimeError)。

        fp8 など torchao のバージョンによって無い config を、初回リクエストのロード時まで
        持ち越して分かりにくく失敗させないため。
        """
        if not self.quantized:
            return
        if self.kitchen:
            try:
                import comfy_kitchen as ck

                backends = ck.list_backends()
                capable = [n for n, b in backends.items()
                           if b.get("available") and "w4a8_int8_linear" in b.get("capabilities", [])]
                if not capable:
                    raise RuntimeError(f"w4a8_int8_linear を提供するバックエンドがありません: {backends}")
                logger.info("comfy-kitchen w4a8 backends: %s", capable)
            except Exception as e:
                raise RuntimeError(
                    f"H3_PRUNED_QUANT={self.name!r} には comfy-kitchen が必要です"
                    f"(pip install comfy-kitchen==0.2.37): {e!r}"
                ) from e
            return
        try:
            self.make_config()
        except (ImportError, AttributeError, TypeError) as e:
            import importlib.metadata

            raise RuntimeError(
                f"H3_PRUNED_QUANT={self.name!r} に必要な torchao の config を作れません"
                f"(torchao {importlib.metadata.version('torchao')}): {e!r}"
            ) from e

    def describe(self, group_size: int) -> str:
        if not self.quantized:
            return "bf16 (no quantization, ~40GB resident, no cache)"
        if self.kitchen:
            return f"{self.meta_quant_config}, cache={self.cache_name}"
        rot = f" + ConvRot(group={group_size})" if self.convrot else " (no ConvRot)"
        return f"{self.meta_quant_config.removesuffix('+convrot')}{rot}, cache={self.cache_name}"


DEFAULT_PRUNED_QUANT = "int8wo"

PRUNED_QUANT_SPECS: dict[str, PrunedQuantSpec] = {
    # 旧既定(2026-10-01 に int8wo へ変更)。ディレクトリ名・state ファイル名・quant_config
    # 文字列は既存キャッシュ transformer_ref_pruned_int8convrot を無効化しないよう旧実装の
    # 値をそのまま使う。
    "int8dyn-convrot": PrunedQuantSpec(
        name="int8dyn-convrot",
        convrot=True,
        cache_name="transformer_ref_pruned_int8convrot",
        state_filename="pruned_int8convrot_state.pt",
        meta_quant_config="Int8DynamicActivationInt8WeightConfig+convrot",
        config_factory=_cfg_int8dyn,
    ),
    "int8wo": PrunedQuantSpec(
        name="int8wo",
        convrot=False,
        cache_name="transformer_ref_pruned_int8wo",
        state_filename="pruned_int8wo_state.pt",
        meta_quant_config="Int8WeightOnlyConfig(version=2, PerRow)",
        config_factory=_cfg_int8wo,
    ),
    "int8wo-convrot": PrunedQuantSpec(
        name="int8wo-convrot",
        convrot=True,
        cache_name="transformer_ref_pruned_int8wo_convrot",
        state_filename="pruned_int8wo_convrot_state.pt",
        meta_quant_config="Int8WeightOnlyConfig(version=2, PerRow)+convrot",
        config_factory=_cfg_int8wo,
    ),
    # comfy-kitchen の W4A8(4bit 重み codebook+group16 fp8 スケール、int8-convrot 活性)。
    # 2026-10-07 マイクロベンチ: H3 形状の大 M GEMM で bf16 比 1.7〜1.9 倍速
    # (torchao tinygemm の int4wo は 5.5 倍遅 — docs/h3-comfy-kitchen-w4a8-20261007.md)。
    "ck-w4a8": PrunedQuantSpec(
        name="ck-w4a8",
        convrot=False,
        cache_name="transformer_ref_pruned_ck_w4a8",
        state_filename="pruned_ck_w4a8_state.pt",
        meta_quant_config="comfy-kitchen asym_w4a8_int8 (bits=4, group16, codebook, convrot256)",
        config_factory=None,
        kitchen=True,
    ),
    "int4wo-convrot": PrunedQuantSpec(
        name="int4wo-convrot",
        convrot=True,
        cache_name="transformer_ref_pruned_int4wo_convrot",
        state_filename="pruned_int4wo_convrot_state.pt",
        meta_quant_config="Int4WeightOnlyConfig(group_size=128, tinygemm)+convrot",
        config_factory=_cfg_int4wo,
    ),
    "int4wo": PrunedQuantSpec(
        name="int4wo",
        convrot=False,
        cache_name="transformer_ref_pruned_int4wo",
        state_filename="pruned_int4wo_state.pt",
        meta_quant_config="Int4WeightOnlyConfig(group_size=128, tinygemm)",
        config_factory=_cfg_int4wo,
    ),
    "fp8": PrunedQuantSpec(
        name="fp8",
        convrot=False,
        cache_name="transformer_ref_pruned_fp8",
        state_filename="pruned_fp8_state.pt",
        meta_quant_config="Float8DynamicActivationFloat8WeightConfig(PerRow)",
        config_factory=_cfg_fp8dyn,
    ),
    "fp8-convrot": PrunedQuantSpec(
        name="fp8-convrot",
        convrot=True,
        cache_name="transformer_ref_pruned_fp8_convrot",
        state_filename="pruned_fp8_convrot_state.pt",
        meta_quant_config="Float8DynamicActivationFloat8WeightConfig(PerRow)+convrot",
        config_factory=_cfg_fp8dyn,
    ),
    "bf16": PrunedQuantSpec(
        name="bf16",
        convrot=False,
        cache_name=None,
        state_filename=None,
        meta_quant_config="bf16",
        config_factory=None,
    ),
}


def get_quant_spec(name: str | None) -> PrunedQuantSpec:
    """`H3_PRUNED_QUANT` の値を検証して spec を返す(未知の値は ValueError)。"""
    key = (name or DEFAULT_PRUNED_QUANT).strip().lower()
    if key not in PRUNED_QUANT_SPECS:
        raise ValueError(
            f"H3_PRUNED_QUANT must be one of {sorted(PRUNED_QUANT_SPECS)}, got {name!r}"
        )
    return PRUNED_QUANT_SPECS[key]


# 旧名の互換(旧既定 int8dyn-convrot の state ファイル名。既定変更後も同じ値を保つ)。
STATE_FILENAME = PRUNED_QUANT_SPECS["int8dyn-convrot"].state_filename


def pruned_snapshot_dir(repo: str, weights: bool = True) -> Path:
    """pruned リポジトリの transformer_ref を含むスナップショットを解決する。

    キャッシュ済みなら即座に返る(snapshot_download は完全なローカルキャッシュに
    対してネットワークを叩かない)。未取得ならここでダウンロードが走る(~38GB)。

    weights=False は量子化済みキャッシュからのロード用: remote code・config 等の
    小物だけを対象にし、**bf16 重みシャード(~36GB)を対象にしない**。重みシャードは
    ディスク節約のためキャッシュ作成後に削除してよい運用(2026-10-05、Z-Image fp32
    削除 = CLAUDE.md 36番と同じ外科手術)にしており、weights=True のパターンで
    解決すると snapshot_download が欠けたシャードを 38GB 再ダウンロードしてしまう。
    """
    from huggingface_hub import snapshot_download

    if weights:
        patterns = ["transformer_ref/*"]
    else:
        patterns = [
            "transformer_ref/config.json",
            "transformer_ref/modeling_minimax_h3_pruned.py",
            "transformer_ref/adaln_affine.safetensors",
            "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
        ]
    path = snapshot_download(repo, allow_patterns=patterns)
    return Path(path)


def load_pruned_module(snapshot: Path):
    """remote code(modeling_minimax_h3_pruned.py)を import してモジュールを返す。"""
    key = str(snapshot)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    src = snapshot / "transformer_ref" / "modeling_minimax_h3_pruned.py"
    spec = importlib.util.spec_from_file_location("modeling_minimax_h3_pruned", str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_minimax_h3_pruned"] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def mark_convrot(transformer, mod, group_size: int) -> int:
    """キャッシュ再ロード後に ConvRot クラスを付け直す(重みは回転済みのまま)。

    enable_convrot()/quantize_8bit() は重みを回転**してから**クラスを差し替えるが、
    キャッシュには回転済みの重みが保存されている。ここでは回転を行わず、
    クラスと groupsize の付与だけを行う。ConvRot 無しの方式で保存したキャッシュに
    対して呼ぶと無言で誤った出力になるので、呼び出し側は `PrunedQuantSpec.convrot`
    で必ずゲートすること。
    """
    names = transformer.convrot_layers()
    lookup = dict(transformer.named_modules())
    for name in names:
        module = lookup[name]
        module.__class__ = mod.MiniMaxH3ConvRotLinear
        module.convrot_groupsize = group_size
    transformer._convrot_layers = names
    return len(names)


class CKW4A8Linear(torch.nn.Linear):
    """comfy-kitchen W4A8 の推論専用 Linear(重みは 4bit パック + スケール群のバッファ)。

    nn.Linear を継承するのは turbo LoRA 適用(`_apply_turbo_lora_*` の
    `isinstance(base_linear, torch.nn.Linear)` と in/out_features 参照)を素通しする
    ため。`weight` パラメータは __init__ 直後に除去する(forward は全面差し替え、
    state_dict にも現れない)。バッファは shape 既知(in/out から決まる)なので
    meta 構築 -> `load_state_dict(assign=True)` のキャッシュ再ロードがそのまま通る。

    注意: `.to(dtype=...)` を親モデルに掛けると浮動小数バッファ(s_rel fp8 /
    s_channel f32 / codebook f32)が巻き込まれて壊れる。pruned 経路は device 移動
    (`.to(device)`)しか行わないので現状安全だが、dtype キャストを足すときは
    このクラスを除外すること。
    """

    GROUP_SIZE = 16

    @staticmethod
    def convrot_groupsize_for(in_features: int) -> int:
        """K(in_features)を割り切る最大の回転グループ。pruned の対象層は全て 256 だが、
        base(非 pruned)には K=2688 など 256 で割り切れない層がある(実機で検出)。
        in_features から決定的に導出するので、キャッシュ再ロードのシェル側でも
        同じ値が再現される(state_dict に持たせる必要がない)。"""
        for g in (256, 128, 64, 32, 16):
            if in_features % g == 0:
                return g
        raise RuntimeError(f"in_features={in_features} は 16 で割り切れず W4A8 化できません")

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.convrot_groupsize = self.convrot_groupsize_for(in_features)
        del self._parameters["weight"]
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device="meta"), requires_grad=False
            )
        self.register_buffer("qdata", torch.empty(out_features, in_features // 2,
                                                  dtype=torch.int8, device="meta"))
        self.register_buffer("s_rel", torch.empty(out_features, in_features // self.GROUP_SIZE,
                                                  dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("s_channel", torch.empty(out_features, dtype=torch.float32, device="meta"))
        self.register_buffer("codebook", torch.empty(16, dtype=torch.float32, device="meta"))

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, device) -> "CKW4A8Linear":
        """bf16 nn.Linear を層単位で GPU 量子化して置換用モジュールを作る(結果は CPU 常駐、
        呼び出し側の最後の `tr.to(device)` で GPU へ移る — `_quantize_plain` と同じ流儀)。"""
        import comfy_kitchen as ck

        w = module.weight.data.to(device)
        qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
            w, group_size=cls.GROUP_SIZE,
            convrot_groupsize=cls.convrot_groupsize_for(module.in_features),
        )
        if correction is not None:
            # 既定引数では None(実機確認済み)。付いて返る条件が増えたら保存形式ごと
            # 見直しが要るので、黙って捨てずに落とす。
            raise RuntimeError("quantize_w4a8_int8_weight が correction を返しました(未対応)")
        new = cls(module.in_features, module.out_features, module.bias is not None)
        new.qdata = qdata.to("cpu")
        new.s_rel = s_rel.to("cpu")
        new.s_channel = s_channel.to("cpu")
        new.codebook = codebook.to("cpu")
        if module.bias is not None:
            new.bias = torch.nn.Parameter(
                module.bias.data.detach().to("cpu", torch.bfloat16), requires_grad=False
            )
        del w
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        shape = x.shape
        y = ck.w4a8_int8_linear(
            x.reshape(-1, shape[-1]),
            self.qdata, self.s_rel, self.s_channel,
            codebook=self.codebook, correction=None, bias=self.bias,
            group_size=self.GROUP_SIZE, convrot_groupsize=self.convrot_groupsize,
            out_dtype=x.dtype,
        )
        return y.reshape(*shape[:-1], y.shape[-1])


def _set_module_by_path(root: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def kitchen_linear_names(model, exclude_substrings) -> list[str]:
    """ck-w4a8 の対象 nn.Linear 名を列挙する(base = 非 pruned transformer 用)。

    除外は torchao の modules_to_not_convert と同じ部分一致セマンティクス
    (`any(sub in name)`) — int8 経路とレシピの除外集合を揃えるため。
    `type(module) is nn.Linear` 判定なのは、既に置換/ラップ済みモジュールを
    二重変換しないため(_quantize_plain と同じ構造保証)。"""
    return [
        name for name, module in model.named_modules()
        if type(module) is torch.nn.Linear and not any(sub in name for sub in exclude_substrings)
    ]


def quantize_kitchen_names(model, names: list[str], device) -> None:
    """名前リストで指定した Linear 群を CKW4A8Linear へ置換する(層単位 GPU 量子化)。"""
    lookup = dict(model.named_modules())
    for name in names:
        module = lookup[name]
        if type(module) is not torch.nn.Linear:
            raise RuntimeError(f"{name} は既に置換済みです({type(module).__name__})")
        _set_module_by_path(model, name, CKW4A8Linear.from_linear(module, device))


def install_kitchen_shells_names(model, names: list[str]) -> int:
    """キャッシュ再ロード用: 指定 Linear 群を meta の CKW4A8Linear 空シェルへ差し替える。"""
    lookup = dict(model.named_modules())
    for name in names:
        module = lookup[name]
        _set_module_by_path(model, name, CKW4A8Linear(
            module.in_features, module.out_features, module.bias is not None))
    return len(names)


def _quantize_kitchen(transformer, device) -> list[str]:
    """ConvRot 対象と同じ 300 層を comfy-kitchen W4A8 の CKW4A8Linear へ置換する。

    `_quantize_plain` と同様に層単位で GPU 量子化する(CPU 上の bf16 40GB モデルに
    対して GPU 一時確保を重み1枚ぶんに抑える)。remote code の ConvRot 機構
    (mark_convrot / _convrot_layers)は一切使わない(kitchen カーネルが回転を内包)。
    """
    names = transformer.convrot_layers()
    quantize_kitchen_names(transformer, names, device)
    return names


def _install_kitchen_shells(transformer) -> int:
    """キャッシュ再ロード用: meta 構築済みモデルの ConvRot 対象層を CKW4A8Linear の
    空シェル(meta、正しい shape)へ差し替える。続く `load_state_dict(assign=True)` が
    qdata/s_rel/s_channel/codebook(/bias)を実体化する。"""
    names = transformer.convrot_layers()
    return install_kitchen_shells_names(transformer, names)


def _quantize_plain(transformer, config, device) -> list[str]:
    """ConvRot を掛けずに、ConvRot 対象と同じ 300 層へ torchao `quantize_` を層単位で適用する。

    remote code `quantize_8bit()` から回転(fold + クラス差し替え)だけを除いた同じ手順。
    層ごとに GPU へ上げて量子化し元の device へ戻すのは、CPU 上の bf16 40GB モデルに対して
    GPU 側の一時確保を重み 1 枚ぶんに抑えるため(Float8 の量子化は CUDA 必須でもある)。
    enable_convrot/quantize_8bit を呼ばないので重みは回転されずクラスも nn.Linear の
    ままになり、`_convrot_layers` も設定しない(convrot_filter 以外は読まない属性)。
    """
    from torchao.quantization import quantize_

    names = transformer.convrot_layers()
    lookup = dict(transformer.named_modules())
    for name in names:
        module = lookup[name]
        if type(module.weight) not in (torch.Tensor, torch.nn.Parameter):
            # torchao の Int8Tensor は `_is_linear` の再量子化ガード対象外で二重量子化を
            # 試みてしまうため、「必ず 1 回だけ」を構造で保証する。
            raise RuntimeError(
                f"{name} は既に量子化済みです({type(module.weight).__name__})。"
                "同じ transformer へ量子化を二度適用しようとしています。"
            )
        home = module.weight.device
        module.to(device or home)
        quantize_(module, config)
        module.to(home)
    return names


def load_pruned_ref_fresh(snapshot: Path, device, group_size: int, quant: str = DEFAULT_PRUNED_QUANT):
    """bf16 を CPU へロード -> 方式に応じて量子化(GPU で層単位) -> GPU 常駐。

    `int8dyn-convrot` のフェーズA probe 実測: ロード 0.3s(mmap)+ 量子化 31.1s +
    to() 4.7s、常駐 19.62GiB。どの方式でも CPU 側に一時 ~40GB の bf16 が載る
    (RAM ガードは呼び出し側)。`bf16` は量子化せずそのまま GPU へ載せる(~40.2GB)。
    """
    spec = get_quant_spec(quant)
    mod = load_pruned_module(snapshot)
    t0 = time.time()
    tr = mod.MiniMaxH3PrunedTransformer3DModel.from_pretrained(
        str(snapshot / "transformer_ref"), torch_dtype=torch.bfloat16
    )
    logger.info("pruned transformer_ref bf16 loaded (CPU) in %.1fs", time.time() - t0)
    t1 = time.time()
    if not spec.quantized:
        logger.info("pruned transformer_ref: quant=%s -> 量子化せず bf16 のまま GPU へ載せます", spec.name)
    elif spec.kitchen:
        names = _quantize_kitchen(tr, device)
        logger.info(
            "pruned kitchen W4A8 quantize done in %.1fs (quant=%s, layers=%d)",
            time.time() - t1, spec.name, len(names),
        )
    elif spec.convrot:
        tr.quantize_8bit(config=spec.make_config(), device=device, group_size=group_size)
        logger.info(
            "pruned quantize_8bit done in %.1fs (quant=%s, convrot layers=%d, group=%d)",
            time.time() - t1, spec.name, len(tr._convrot_layers), group_size,
        )
    else:
        names = _quantize_plain(tr, spec.make_config(), device)
        logger.info(
            "pruned plain quantize_ done in %.1fs (quant=%s, layers=%d, no convrot)",
            time.time() - t1, spec.name, len(names),
        )
    tr = tr.to(device)
    return tr


def save_pruned_ref_cache(transformer, tmp_dir: Path, quant: str = DEFAULT_PRUNED_QUANT) -> None:
    """量子化済み pruned transformer_ref をキャッシュへ保存する。

    `save_pretrained()`(safetensors)は使えない: remote code の `quantize_8bit()` は
    torchao `quantize_()` を生で呼ぶため hf_quantizer が付いておらず、safetensors の
    共有テンソル判定(`storage_ptr`)が torchao のテンソルサブクラスで
    `RuntimeError: Attempted to access the data pointer on an invalid python storage`
    になる(2026-10-01 実機で確認。非 pruned int8 は diffusers `TorchAoConfig` 経由で
    hf_quantizer が直列化を仲介するため safetensors で保存できる — この差が原因)。
    代わりに state_dict を torch.save(pickle)で保存する。persistent バッファ
    (table / folded_bias / adaln_basis / adaln_mean)は全て state_dict に含まれ、
    非 persistent バッファは LoRA 適用後にのみ生じる(キャッシュ時点では存在しない)
    ため、この方式で過不足なく復元できる。Int8Tensor / Float8Tensor /
    LinearActivationQuantizedTensor いずれも pickle 往復後の forward が bit 一致する
    ことを確認済み(2026-10-01、小規模 Linear)。
    """
    spec = get_quant_spec(quant)
    if not spec.cacheable:
        raise RuntimeError(f"H3_PRUNED_QUANT={spec.name!r} はキャッシュ対象外です(呼び出し側のバグ)")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_config(str(tmp_dir))
    sd = transformer.state_dict()
    torch.save(sd, str(tmp_dir / spec.state_filename))


def load_pruned_ref_from_cache(
    cache_dir: Path, snapshot: Path, device, group_size: int, quant: str = DEFAULT_PRUNED_QUANT
):
    """量子化済みキャッシュから読み、方式が ConvRot 系ならクラスを付け直して返す。失敗時は例外。

    from_config(meta)-> torch.load -> load_state_dict(assign=True) の順で、
    bf16 のランダム初期化(40GB)を踏まずに量子化済みテンソルを直接実体化する。
    torch.load は weights_only=False(torchao テンソルサブクラスの pickle 復元に必要。
    このキャッシュは本プロセス自身が書いたローカルファイルのみ)。
    キャッシュが現在の方式で作られたものであることは、呼び出し側(runner)が
    meta.json の完全一致で保証する(方式ごとにディレクトリ名も quant_config も異なる)。
    """
    from accelerate import init_empty_weights

    spec = get_quant_spec(quant)
    if not spec.cacheable:
        raise RuntimeError(f"H3_PRUNED_QUANT={spec.name!r} はキャッシュ対象外です(呼び出し側のバグ)")
    mod = load_pruned_module(snapshot)
    t0 = time.time()
    cls = mod.MiniMaxH3PrunedTransformer3DModel
    config = cls.load_config(str(cache_dir))
    with init_empty_weights():
        tr = cls.from_config(config)
    if spec.kitchen:
        n_shell = _install_kitchen_shells(tr)
        logger.info("kitchen W4A8 shells installed for cache reload: %d layers", n_shell)
    sd = torch.load(str(cache_dir / spec.state_filename), map_location="cpu", weights_only=False)
    tr.load_state_dict(sd, assign=True)
    tr = tr.to(device)
    n = mark_convrot(tr, mod, group_size) if spec.convrot else 0
    logger.info(
        "pruned transformer_ref loaded from prequantized cache in %.1fs "
        "(%s, quant=%s, convrot re-marked=%d)", time.time() - t0, cache_dir, spec.name, n,
    )
    return tr


# ---- H3_PRUNED_COMPILE: ConvRot 層の torch.compile(2026-10-05 probe) -----------------------
#
# 動機: torchao 0.17 の int8 動的量子化 eager 経路は per-token 量子化/逆量子化が未融合で
# bf16 GEMM の 2〜3 倍遅い。torch.compile で融合すると単層 3.4〜3.6 倍速くなる(マイクロベンチ)。
# 対象は「ConvRot Linear の forward(入力回転 + 量子化 GEMM)」だけ。LoRA ラッパー
# (_TurboLoRALinear)・attention(sage)・FBC フックは compile に巻き込まない。
#
# remote code の `_rotate` は module 辞書 `_HADAMARD_CACHE`(キーは (size, str(device), dtype))
# を参照しており、dynamo が辞書アクセスと str(device) で graph break する。同値の
# バッファ版(Hadamard 行列を各層の非 persistent buffer として持つ)に差し替える。
# 演算は matmul(reshape(-1, K/g, g), H) -> reshape と _rotate と完全に同一(bit 一致を
# `check_rotate_bit_exact()` で確認できる)。remote code ファイルは書き換えず、
# モジュールの __class__ だけをサブクラスへ差し替える(PatchifyLinear と同じ流儀)。
# state_dict のキーは変わらず(buffer は persistent=False)、キャッシュ形式は不変。

_COMPILABLE_CLS: dict[int, type] = {}


def _compilable_cls(mod):
    key = id(mod.MiniMaxH3ConvRotLinear)
    if key in _COMPILABLE_CLS:
        return _COMPILABLE_CLS[key]
    import torch.nn.functional as F

    base_cls = mod.MiniMaxH3ConvRotLinear

    class MiniMaxH3ConvRotLinearBuf(base_cls):
        """ConvRot Linear の buffer 版(compile 可能)。`_had` は [g, g] の Hadamard(入力 dtype)。"""

        def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
            had = self._had
            if input.dtype != had.dtype:
                # 想定外の dtype(通常 bf16 のみ)。remote code の経路へ退避(graph break するが正しい)。
                return base_cls.forward(self, input)
            shape = input.shape
            g = self.convrot_groupsize
            x = torch.matmul(input.reshape(-1, shape[-1] // g, g), had).reshape(shape)
            return F.linear(x, self.weight, self.bias)

    _COMPILABLE_CLS[key] = MiniMaxH3ConvRotLinearBuf
    return MiniMaxH3ConvRotLinearBuf


def patch_convrot_buffers(transformer, mod, dtype=torch.bfloat16) -> int:
    """全 ConvRot 層を buffer 版クラスへ差し替える(重みは触らない)。差し替えた層数を返す。"""
    cls = _compilable_cls(mod)
    n = 0
    shared: dict[tuple, torch.Tensor] = {}
    for module in transformer.modules():
        if not isinstance(module, mod.MiniMaxH3ConvRotLinear):
            continue
        g = module.convrot_groupsize
        dev = module.weight.device
        key = (g, str(dev), dtype)
        if key not in shared:
            # remote code と同一の関数・同一の引数で作る(= _rotate が使う行列と bit 一致)。
            shared[key] = mod._hadamard(g, dev, dtype).clone()
        module.__class__ = cls
        module.register_buffer("_had", shared[key], persistent=False)
        n += 1
    return n


def check_rotate_bit_exact(mod, group_size: int = 256, device="cuda", dtype=torch.bfloat16) -> bool:
    """buffer 版の回転が remote code の `_rotate` と torch.equal で一致するか(単体確認用)。"""
    torch.manual_seed(0)
    ok = True
    had = mod._hadamard(group_size, device, dtype).clone()
    for shape in [(1, group_size * 4), (777, group_size * 21), (3, 5, group_size * 2)]:
        x = torch.randn(*shape, device=device, dtype=dtype)
        ref = mod._rotate(x, group_size)
        s = x.shape
        new = torch.matmul(x.reshape(-1, s[-1] // group_size, group_size), had).reshape(s)
        ok = ok and torch.equal(ref, new)
    return ok


def compile_convrot_layers(transformer, mod, *, dynamic: bool = True, recompile_limit: int = 64) -> int:
    """全 ConvRot 層(buffer 版)を `module.compile()` する(遅延: 初回 forward でコンパイル)。

    torch.compile は dynamic=True(トークン数 M がリクエストごとに変わるため)。
    dynamo の per-code キャッシュ上限は層の重み shape の種類 × 動的 shape で足りなくなり得る
    ので引き上げる(プロセス全体の dynamo 設定だが、他に compile を使う箇所は無い)。
    inductor / torch 全体の matmul precision 設定には触らない。
    """
    import torch._dynamo

    n = patch_convrot_buffers(transformer, mod)
    cfg = torch._dynamo.config
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, attr) and getattr(cfg, attr) < recompile_limit:
            setattr(cfg, attr, recompile_limit)
    if hasattr(cfg, "accumulated_recompile_limit"):
        cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, recompile_limit * 8)
    if hasattr(cfg, "accumulated_cache_size_limit"):
        cfg.accumulated_cache_size_limit = max(cfg.accumulated_cache_size_limit, recompile_limit * 8)
    cls = _compilable_cls(mod)
    m = 0
    for module in transformer.modules():
        if type(module) is cls:
            module.compile(dynamic=dynamic)
            m += 1
    logger.info("H3_PRUNED_COMPILE: ConvRot buffer-patched %d layers, torch.compile(dynamic=%s) on %d", n, dynamic, m)
    return m


def compile_turbo_wrappers(transformer, wrapper_cls, *, dynamic: bool = True) -> int:
    """(実験, H3_PRUNED_COMPILE=2) turbo LoRA ラッパーのうち base が buffer 版 ConvRot のものを compile する。

    `y = base(x) + scale * B(A(x))` を 1 グラフにして mul/add を融合する。ラッパーは `.enabled`
    (python bool)を forward で見るため、トグルで 1 回だけ再コンパイルされる。
    """
    n = 0
    for module in transformer.modules():
        if isinstance(module, wrapper_cls) and type(module.base) in _COMPILABLE_CLS.values():
            module.compile(dynamic=dynamic)
            n += 1
    logger.info("H3_PRUNED_COMPILE=2: turbo LoRA wrappers compiled on %d layers", n)
    return n


def compile_transformer_blocks(transformer, *, dynamic: bool = True) -> int:
    """(probe, H3_PRUNED_COMPILE=3) transformer の各ブロック (50) を compile する。

    ブロック内の ConvRot / turbo LoRA ラッパーは既に compile 済み (level 1/2) で、外側の compile に
    inline される。目的は eager に残る norms・AdaLN 変調 (index_select + mul/add)・残差・rope・swiglu の融合。
    attention の本体 (`dispatch_attention_fn`: sage の CUDA 拡張) は dynamo では trace できないので
    `torch._dynamo.disable` で包み、きれいな graph break にする。
    """
    import torch._dynamo
    from diffusers.models.transformers import transformer_minimax_h3 as tm

    if not getattr(tm.dispatch_attention_fn, "_h3_dynamo_disabled", False):
        wrapped = torch._dynamo.disable(tm.dispatch_attention_fn)
        wrapped._h3_dynamo_disabled = True
        tm.dispatch_attention_fn = wrapped
    n = 0
    for block in transformer.transformer_blocks:
        block.compile(dynamic=dynamic)
        n += 1
    logger.info("H3_PRUNED_COMPILE=3: %d transformer blocks compiled (dynamic=%s)", n, dynamic)
    return n
