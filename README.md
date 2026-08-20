# TideSpec

[日本語版](#日本語版) | [English](#english)

<!-- demo video: after push, drag & drop demo.mp4 into the README web editor here and keep the generated user-attachments URL -->

Regime-adaptive dictionary drafting layered on SGLang's DFlash2 speculative decoding. Experimental patch, tuned for a single RTX 5090.

---

## English

### Features

- 🌊 **Regime-adaptive drafting** — switches the draft source between DFlash2 and a C++ trie dictionary depending on how predictable the current output is
- 📖 **Dictionary arm** — depth-18 sliding-window trie built from the session's own tokens; strongest on edits, regeneration, and boilerplate
- 🎯 **Shadow scoring** — every dictionary proposal is scored on every step, even when not fired, so the controller always knows the dictionary's current hit rate
- 📈 **AIMD firing control** — success halves the firing bar, two consecutive failures reset it
- ✅ **Quality-verified** — every proposed token is verified by the target model before acceptance; checked ON vs OFF on HumanEval / GSM8K below
- 📊 **Live dashboard** (`dashboard.py`) — streams test cases against the server and shows live tok/s with the server log as evidence

### Benchmarks

Environment: RTX 5090 32GB / Windows 11 + WSL2 / SGLang / Qwen3.8-27B NVFP4 with DFlash2 draft. All runs at temp 0, reasoning_effort=low.

Speed (from server logs):

| Configuration | tok/s |
|---|---|
| Pure autoregressive (no speculation) | 74 |
| DSpark | 141 |
| DFlash2 | 197 |
| TideSpec v4.6, official 9-case mean | 167 (+3.8% vs DFlash2, 6 wins / 2 ties / 1 loss) |
| TideSpec v4.6, edit workloads | 339–358 |
| TideSpec v4.6, peak (live measurement) | 518 |

Quality, ON vs OFF (details in [results/](results/)):

| Benchmark | OFF | ON |
|---|---|---|
| HumanEval pass@1 (164 problems, execution-graded) | 0.909 | 0.902 (1 problem, within noise) |
| GSM8K (200 problems) | 0.985 | 0.985 |

### Prerequisites

- Docker with GPU support (tested on Docker Desktop + WSL2)
- Upstream image `lmsysorg/sglang`
- Models are pulled from Hugging Face automatically: `RadixArk/Qwen3.8-27B-NVFP4` (target) + `incoai/Qwen3.8-27B-DFlash2` (draft)

### Quick Start

1. Build the image chain (see header comments in each Dockerfile):

```bash
docker build -f docker/Dockerfile.dflash2  -t sglang:dflash2 .
docker build -f docker/Dockerfile.abortfix -t sglang:dflash2-fixed .
docker build -f docker/Dockerfile.fusion   -t sglang:dflash2-fusion .
```

2. Launch the server:

```bash
run/run-tidespec.bat
```

`SGLANG_DFLASH_NGRAM_FUSION=1` is the opt-in switch — set it to `0` for plain DFlash2 with the same image.

3. Run benchmarks from `bench/` (set `HF_CACHE` to your host huggingface cache):

```bash
HF_CACHE="$USERPROFILE/.cache/huggingface" bash bench/quality_suite.sh
```

### How it works (v4.6)

Three instruments plus two actuators; details in [docs/tidespec-design.md](docs/tidespec-design.md).

- Shadow scoring: dictionary proposals are scored every step, maintaining an EMA of dictionary accuracy relative to the model
- Full-match signal: detects the instant the recent token sequence exactly matches a dictionary path
- AIMD firing control: success halves the firing bar, two consecutive failures reset it
- Dictionary drafting: on qualifying steps the draft is replaced with the trie continuation
- Canonical mask filler: block-verification gaps are filled with canonical tokens

### Repository layout

```
patches/   tidespec.patch (diff against upstream sha 28198c8), patched full file, apply script
docker/    Dockerfile chain (upstream -> dflash2 -> abortfix -> fusion)
run/       server launch scripts (run-tidespec.bat / run-dflash2.bat)
bench/     speed bench, HumanEval+GSM8K quality suite
results/   quality verification reports and raw speed data
docs/      design document, research report
```

### Troubleshooting

- **Bogus OOM while loading the draft on WSL2** — `--weight-loader-disable-mmap` is mandatory: on this machine, mmap'd draft loading consumed ~6GB of extra GPU address space (apparently GPU-PV aperture accounting; untested on native Linux)
- **`AssertionError: Can not alloc mamba cache` under rapid multi-turn load** — upstream issue; benchmarks work around it with `--disable-radix-cache`
- **`run_eval` silently retries forever** — pass `--host 127.0.0.1` without a scheme; the harness prepends `http://` itself

### ⚠️ Important Notice

This is an experimental patch for research purposes, pinned to upstream sglang sha `28198c8` (requires NVFP4-target lm_head support, [PR #35496](https://github.com/sgl-project/sglang/pull/35496) or later) and tuned for a single stream (`--max-running-requests 1`) on one RTX 5090. It is not production software.

This repository is a personal project and does not accept pull requests.

`dashboard.py` binds to `0.0.0.0` with no authentication: anyone on the same network can submit prompts to your server and view its logs. Run it on trusted networks only.

### Authors & Contributors

- **m-sigepon** - Project creator
- **Claude Code** - Code development & implementation

This tidespec was developed using Claude Code.

### License

Apache-2.0. Patched code derives from [sgl-project/sglang](https://github.com/sgl-project/sglang) (Apache-2.0).

### Acknowledgments

- [SGLang](https://github.com/sgl-project/sglang) — the serving framework this patches
- [DFlash2](https://inco.ai/blog/dflash2/) — the block-diffusion draft model TideSpec builds on
- [human-eval](https://github.com/openai/human-eval) — HumanEval grading

---

## 日本語版

### 特徴

- 🌊 **レジーム適応ドラフティング** — 出力の予測しやすさに応じて、ドラフト元を DFlash2 と C++ trie 辞書の間で切り替える
- 📖 **辞書アーム** — セッション自身のトークンから作る深さ18のスライディングウィンドウ trie。編集・再生成・定型出力に強い
- 🎯 **影スコアリング** — 発火しないステップでも辞書提案を毎回採点し、辞書の現在の的中率を常に把握する
- 📈 **AIMD 発火制御** — 成功で発火バーを半減、2連敗でリセット
- ✅ **品質検証済み** — 提案トークンは毎回ターゲットモデルが検証してから採用。HumanEval / GSM8K の ON/OFF 比較で確認
- 📊 **ライブダッシュボード**（`dashboard.py`）— テストケースをサーバーに流し、サーバーログを証跡として tok/s をリアルタイム表示

### ベンチマーク

環境: RTX 5090 32GB / Windows 11 + WSL2 / SGLang / Qwen3.8-27B NVFP4 + DFlash2 draft。全て temp 0、reasoning_effort=low。

速度（サーバーログ実測）:

| 条件 | tok/s |
|---|---|
| 純粋 AR（投機なし） | 74 |
| DSpark | 141 |
| DFlash2 | 197 |
| TideSpec v4.6 公式9ケース平均 | 167（DFlash2 比 +3.8%、6勝2分1敗） |
| TideSpec v4.6 編集ワークロード | 339–358 |
| TideSpec v4.6 ピーク（ライブ計測） | 518 |

品質（ON/OFF 比較、詳細は [results/](results/)）:

| ベンチ | OFF | ON |
|---|---|---|
| HumanEval pass@1（164問、コード実行採点） | 0.909 | 0.902（1問差、誤差内） |
| GSM8K（200問） | 0.985 | 0.985 |

### 前提

- GPU 対応の Docker（Docker Desktop + WSL2 で検証）
- 上流イメージ `lmsysorg/sglang`
- モデルは Hugging Face から自動取得: `RadixArk/Qwen3.8-27B-NVFP4`（ターゲット）+ `incoai/Qwen3.8-27B-DFlash2`（ドラフト）

### クイックスタート

1. イメージチェーンをビルド（各 Dockerfile 冒頭のコメント参照）:

```bash
docker build -f docker/Dockerfile.dflash2  -t sglang:dflash2 .
docker build -f docker/Dockerfile.abortfix -t sglang:dflash2-fixed .
docker build -f docker/Dockerfile.fusion   -t sglang:dflash2-fusion .
```

2. サーバー起動:

```bash
run/run-tidespec.bat
```

環境変数 `SGLANG_DFLASH_NGRAM_FUSION=1` が opt-in スイッチ。`0` にすると同じイメージで素の DFlash2 になる。

3. ベンチは `bench/` から（`HF_CACHE` でホスト側キャッシュを指定）:

```bash
HF_CACHE="$USERPROFILE/.cache/huggingface" bash bench/quality_suite.sh
```

### 仕組み（v4.6）

計測 3 点 + 作動 2 点。詳細は [docs/tidespec-design.md](docs/tidespec-design.md)。

- 影スコアリング: 辞書提案を毎ステップ採点し、辞書の的中率をモデル比の EMA で持つ
- 完全一致シグナル: 直近トークン列が辞書と完全一致した瞬間を検出
- AIMD 発火制御: 成功で発火バーを半減、2連敗でリセット
- 辞書ドラフト: 条件を満たしたステップでドラフトを trie の続きに差し替え
- 正準マスク充填: ブロック検証の穴埋めに正準トークンを使う

### リポジトリ構成

```
patches/   tidespec.patch（上流 sha 28198c8 への diff）、パッチ適用済み全文、適用スクリプト
docker/    イメージ 3 段（upstream → dflash2 → abortfix → fusion）の Dockerfile
run/       サーバー起動バッチ（run-tidespec.bat / run-dflash2.bat）
bench/     速度ベンチ、HumanEval+GSM8K 品質スイート
results/   品質検証レポートと速度生データ
docs/      設計文書、研究レポート
```

### トラブルシューティング

- **WSL2 でドラフトロード中に偽 OOM** — `--weight-loader-disable-mmap` が必須。この環境では mmap ロードが約6GB余分に GPU アドレス空間を消費した（GPU-PV の計上によるものとみられる。ネイティブ Linux は未検証）
- **マルチターン高速連発で `AssertionError: Can not alloc mamba cache`** — 上流の問題。ベンチでは `--disable-radix-cache` で回避
- **`run_eval` が無言でリトライし続ける** — `--host 127.0.0.1` をスキームなしで渡す（ハーネスが `http://` を自分で付ける）

### ⚠️ 注意

研究目的の実験パッチです。上流 sglang sha `28198c8` にピン留めしており（NVFP4 ターゲットの lm_head 対応、[PR #35496](https://github.com/sgl-project/sglang/pull/35496) 以降が前提）、RTX 5090 一枚・単一ストリーム（`--max-running-requests 1`）向けのチューニングです。プロダクション用ソフトウェアではありません。

このリポジトリは個人プロジェクトのため、プルリクエストは受け付けておりません。

`dashboard.py` は認証なしで `0.0.0.0` にバインドします。同一ネットワーク上の誰でもサーバーへプロンプトを送信でき、ログも閲覧できます。信頼できるネットワークでのみ使用してください。

### 作者・貢献者

- **m-sigepon** - プロジェクト作成者
- **Claude Code** - コード開発・実装

この tidespec は Claude Code を使用して開発されました。

### License

Apache-2.0。パッチ対象コードは [sgl-project/sglang](https://github.com/sgl-project/sglang)（Apache-2.0）由来。

### Acknowledgments

- [SGLang](https://github.com/sgl-project/sglang) — パッチ対象のサービングフレームワーク
- [DFlash2](https://inco.ai/blog/dflash2/) — TideSpec の土台であるブロック拡散ドラフトモデル
- [human-eval](https://github.com/openai/human-eval) — HumanEval の採点
