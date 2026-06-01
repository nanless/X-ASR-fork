#!/bin/bash
# X-ASR 语音输入法 启动脚本
set -e
cd "$(dirname "$0")"

VENV="../x-asr-live-demo/.venv"

if [ ! -d "$VENV" ]; then
    echo ">>> 创建虚拟环境…"
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -r ../x-asr-live-demo/requirements.txt
    pip install pyobjc-framework-Quartz
else
    source "$VENV/bin/activate"
fi

python -c "import Quartz" 2>/dev/null || pip install pyobjc-framework-Quartz

echo ">>> 启动 X-ASR 语音输入法…"
echo "  ⌘⇧Space  开始/停止录音"
echo "  文字实时上屏到焦点应用"
python voice_ime.py "$@"
