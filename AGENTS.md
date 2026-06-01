# AGENTS.md — X-ASR

## 项目简介

基于 icefall/k2、Zipformer 和 sherpa-onnx 的流式语音识别（ASR）模型。当前发布版本为 **X-ASR-zh-en**（中英双语，约 1.6 亿参数，约 100 万小时训练数据）。

## 仓库结构

- `X-ASR-zh-en/deployment/` — sherpa-onnx WebSocket 服务端/客户端 + 本地实时演示。可直接运行的部署代码。
- `X-ASR-zh-en/zipformer/` — icefall 训练配方、导出脚本、PyTorch 检查点和 BPE 分词器。
- `assets/` — README 使用的图片和演示素材。

**不要混用** PyTorch 检查点（`zipformer/checkpoint/`）与 ONNX 部署产物（`deployment/models/`），它们来自不同的导出路径。

## Git LFS

ONNX 模型文件（`.onnx`）和媒体文件通过 **Git LFS** 跟踪。克隆仓库或切换分支后必须执行：

```bash
git lfs install
git lfs pull
```

否则 `.onnx` 文件会是极小的指针桩，推理会静默失败或崩溃。

## 部署环境搭建

部署使用 `X-ASR-zh-en/deployment/` 下的独立虚拟环境：

```bash
cd X-ASR-zh-en/deployment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖：`numpy`、`websockets`、`soundfile`、`librosa`、`sherpa-onnx`。

实时演示（`x-asr-live-demo/`）有独立的 venv 和依赖：

```bash
cd X-ASR-zh-en/deployment/x-asr-live-demo
pip install -r requirements.txt          # 核心依赖（silero/energy VAD）
pip install -r requirements-firered.txt  # 可选 FireRedVAD
./download_models.sh                     # 下载 ASR + VAD 模型权重
```

## 启动 WebSocket 服务端

必须提供来自**同一模型目录**的四个文件。encoder、decoder、joiner 和 tokens.txt 必须匹配。

```bash
# 在 X-ASR-zh-en/deployment/ 下执行
python infer_and_client/sherpa_streaming_server.py \
  --host 0.0.0.0 --port 8766 \
  --tokens models/chunk-160ms-model/tokens.txt \
  --encoder models/chunk-160ms-model/encoder-160ms.onnx \
  --decoder models/chunk-160ms-model/decoder-160ms.onnx \
  --joiner models/chunk-160ms-model/joiner-160ms.onnx \
  --provider cpu \
  --sample-rate 16000 --feature-dim 80 \
  --decoding-method greedy_search \
  --model-type zipformer2 \
  --text-format none
```

默认 `--text-format` 为 `lower`（英文小写输出）。使用 `none` 保留模型原始大小写。

四种 chunk 模型变体：`chunk-160ms-model`、`chunk-480ms-model`、`chunk-960ms-model`、`chunk-1920ms-model`。chunk 越小延迟越低，chunk 越大输出越稳定。

## 运行 WebSocket 客户端

```bash
python infer_and_client/sherpa_streaming_client.py \
  --server-uri ws://127.0.0.1:8766 \
  --wav /path/to/test.wav \
  --chunk-ms 100 --simulate-realtime 1
```

音频必须是 **16kHz 单声道 int16 PCM**。客户端会自动重采样。

## 运行实时演示

```bash
cd X-ASR-zh-en/deployment/x-asr-live-demo
python live_asr.py --vad firered --preroll 1.0  # 麦克风 + FireRedVAD
python live_asr.py --wav test.wav               # 文件输入
python live_asr.py --provider coreml            # Apple Silicon 加速
```

需要先运行 `./download_models.sh`。模型下载到 `models/asr/` 和 `models/silero_vad.onnx`。

## 训练 / 解码（zipformer/）

训练依赖 icefall 框架和多 GPU 环境。检查点与数据目录的映射关系是严格的：

| 检查点 | 数据目录 |
|---|---|
| `checkpoint/pretrained.pt` | `data/lang_5000/` |
| `checkpoint/fintuned_with_punctuation.pt` | `data/lang_5000_with_punctuation/` |

不要混用不匹配的检查点和分词器。完整训练/解码/导出命令见 `zipformer/README.md`。

## WebSocket 协议

客户端与服务端之间的极简流式协议：

1. 客户端发送 JSON：`{"type": "start", "sample_rate": 16000}`
2. 客户端流式发送二进制 int16 PCM 音频块
3. 服务端返回 JSON：`{"type": "partial", "text": "..."}`
4. 客户端发送 JSON：`{"type": "end"}`
5. 服务端返回 JSON：`{"type": "final", "text": "...", "first_partial_latency": 0.42}`

重置：`{"type": "reset"}` → `{"type": "reset_ok"}`

## 关键文件

- `deployment/infer_and_client/sherpa_streaming_infer.py` — 核心 `SherpaStreamingASR` 封装 + 文本格式化
- `deployment/infer_and_client/sherpa_streaming_server.py` — WebSocket 服务端
- `deployment/infer_and_client/sherpa_streaming_client.py` — WAV 测试客户端
- `deployment/x-asr-live-demo/live_asr.py` — 本地麦克风/VAD/ASR 管线
- `zipformer/train.py`、`finetune.py`、`decode.py`、`streaming_decode.py` — 训练配方
- `zipformer/export-onnx.py`、`export-onnx-streaming.py` — ONNX 导出脚本

## 注意事项

- 服务端默认 `--text-format` 为 `lower`，会将所有英文输出转为小写。使用 `--text-format none` 保留模型原始大小写。
- 服务端默认模型路径硬编码为作者机器的绝对路径，**必须**显式传入 `--tokens`、`--encoder`、`--decoder`、`--joiner` 参数。
- 实时演示的 FireRedVAD 是可选的；缺失时会静默回退到 silero VAD。
- `--enable-endpoint-detection` 默认为 `0`（关闭）。实时演示中的端点检测由 VAD 处理，而非 sherpa-onnx 内置端点检测。
- 训练脚本假设多 GPU 环境（`--world-size 8`）且依赖本仓库未包含的 icefall/k2 依赖。
