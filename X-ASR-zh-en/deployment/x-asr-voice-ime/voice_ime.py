#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X-ASR Voice IME — Mac 本地语音输入法
====================================
悬浮窗 + 全局热键(Cmd+Shift+Space) + 流式ASR + 流式文字上屏

用法:
    python voice_ime.py
    python voice_ime.py --hotwords-file hotwords.txt
"""

import argparse
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
import sherpa_onnx
import sounddevice as sd

# ── 常量 ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
VAD_WINDOW = 512  # 32ms @ 16k
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CJK 规范化 ──────────────────────────────────────────────────────────────
_CJK = r"㐀-䶿一-鿿豈-﫿"
_CJK_PUNCT = re.escape("，。！？；：、（）《》〈〉【】「」『』""''")
_ASCII_PUNCT = re.escape(",.!?;:%)]}")


def normalize_cjk(text):
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"\s+(?=[{_ASCII_PUNCT}])", "", text)
    return text


# ── BPE vocab 生成 (热词需要) ────────────────────────────────────────────────
def _make_bpe_vocab(tokens_file):
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="bpe_vocab_")
    with open(tokens_file) as fin, open(fd, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 2:
                fout.write(f"{parts[0]} 0.0\n")
    return path


# ── 文字注入 (剪贴板 + Cmd+V，兼容中英文) ────────────────────────────────────
def _get_clipboard():
    """读取当前剪贴板内容"""
    try:
        r = subprocess.run(["pbpaste"], capture_output=True, timeout=2)
        return r.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _set_clipboard(text):
    """写入剪贴板"""
    try:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), timeout=2)
    except Exception:
        pass


def _paste():
    """发送 Cmd+V"""
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)


def send_backspaces(n):
    """发送 n 个退格键删除已上屏文字"""
    if n <= 0:
        return
    script = f'tell application "System Events" to repeat {n} times to key code 51 end repeat'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)


def inject_text(text):
    """通过剪贴板粘贴将文字注入焦点应用（支持中英文）"""
    if not text:
        return
    old_clip = _get_clipboard()
    _set_clipboard(text)
    time.sleep(0.02)
    _paste()
    time.sleep(0.02)
    _set_clipboard(old_clip)  # 恢复原剪贴板


# ── ASR 引擎 (带流式回调) ────────────────────────────────────────────────────
class ASREngine:
    def __init__(self, model_dir, hotwords_file="", hotwords_score=1.5):
        tokens = os.path.join(model_dir, "tokens.txt")
        enc = os.path.join(model_dir, "encoder-480ms.onnx")
        dec = os.path.join(model_dir, "decoder-480ms.onnx")
        join = os.path.join(model_dir, "joiner-480ms.onnx")
        for f in [tokens, enc, dec, join]:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"模型文件缺失: {f}")

        decoding = "greedy_search"
        bpe_vocab = ""
        modeling_unit = ""
        if hotwords_file:
            decoding = "modified_beam_search"
            modeling_unit = "bpe"
            bpe_vocab = _make_bpe_vocab(tokens)

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens, encoder=enc, decoder=dec, joiner=join,
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method=decoding, provider="cpu", model_type="zipformer2",
            enable_endpoint_detection=False,
            hotwords_file=hotwords_file, hotwords_score=hotwords_score,
            modeling_unit=modeling_unit, bpe_vocab=bpe_vocab,
        )

        cfg = sherpa_onnx.VadModelConfig()
        vad_path = os.path.join(SCRIPT_DIR, "..", "x-asr-live-demo", "models", "silero_vad.onnx")
        if not os.path.isfile(vad_path):
            raise FileNotFoundError(f"VAD 模型缺失: {vad_path}")
        cfg.silero_vad.model = vad_path
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.7
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.window_size = VAD_WINDOW
        cfg.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

        self.audio_q = queue.Queue()
        self.is_recording = False
        self._stop_event = threading.Event()

        # 流式状态 — GUI 轮询这些
        self.partial_text = ""      # 当前 partial（未定稿）
        self.final_text = ""        # 最新一段 final
        self.partial_changed = False
        self.final_changed = False

    def start(self):
        self.is_recording = True
        self._stop_event.clear()
        self.vad.reset()
        self.partial_text = ""
        self.final_text = ""
        self.partial_changed = False
        self.final_changed = False
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self._sd = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=VAD_WINDOW, callback=self._audio_cb,
        )
        self._sd.start()

    def stop(self):
        self._stop_event.set()
        self.is_recording = False
        if hasattr(self, "_sd"):
            self._sd.stop()
            self._sd.close()

    def _audio_cb(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy())

    def _process_loop(self):
        active = False
        preroll = []
        PREROLL = max(1, int(0.7 * SAMPLE_RATE / VAD_WINDOW))

        while not self._stop_event.is_set():
            try:
                w = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self.vad.accept_waveform(w)
            speech = self.vad.is_speech_detected()

            # 语音开始 → 新句子
            if speech and not active:
                active = True
                self.stream = self.recognizer.create_stream()
                for pw in preroll:
                    self.stream.accept_waveform(SAMPLE_RATE, pw)

            # 语音中 → 持续解码
            if active:
                self.stream.accept_waveform(SAMPLE_RATE, w)
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)
                p = normalize_cjk(self.recognizer.get_result(self.stream))
                if p != self.partial_text:
                    self.partial_text = p
                    self.partial_changed = True

            # 语音结束 → final
            if active and not speech:
                self.stream.accept_waveform(SAMPLE_RATE, np.zeros(int(1.0 * SAMPLE_RATE), dtype="float32"))
                self.stream.input_finished()
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)
                final = normalize_cjk(self.recognizer.get_result(self.stream))
                if final.strip():
                    self.final_text = final
                    self.final_changed = True
                self.partial_text = ""
                self.partial_changed = True
                active = False

            preroll.append(w)
            if len(preroll) > PREROLL:
                preroll.pop(0)
            while not self.vad.empty():
                self.vad.pop()

        # 录音结束时 flush 最后一段
        if active and hasattr(self, "stream") and self.stream:
            self.stream.accept_waveform(SAMPLE_RATE, np.zeros(int(1.0 * SAMPLE_RATE), dtype="float32"))
            self.stream.input_finished()
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            final = normalize_cjk(self.recognizer.get_result(self.stream))
            if final.strip():
                self.final_text = final
                self.final_changed = True
            self.partial_text = ""
            self.partial_changed = True


# ── GUI ──────────────────────────────────────────────────────────────────────
class VoiceIMEApp:
    def __init__(self, asr_engine):
        self.asr = asr_engine
        self.root = tk.Tk()
        self.root.title("X-ASR 语音输入法")
        self.root.geometry("400x340+100+100")
        self.root.attributes("-topmost", True)
        self.root.resizable(True, True)

        # 流式上屏状态
        self._typed_len = 0      # 已经上屏到焦点应用的 partial 长度
        self._final_buf = ""     # 已确认的 final 文本

        self._build_ui()
        self._event_tap = None
        self._setup_global_hotkey()
        self._poll_asr()  # 启动 ASR 状态轮询

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=10, pady=6)
        frame.pack(fill=tk.BOTH, expand=True)

        # 标题 + 状态
        top = tk.Frame(frame)
        top.pack(fill=tk.X)
        tk.Label(top, text="🎙️ X-ASR 语音输入法", font=("", 14, "bold")).pack(side=tk.LEFT)
        self.status_label = tk.Label(top, text="就绪", font=("", 10), fg="gray")
        self.status_label.pack(side=tk.RIGHT)

        # 录音按钮
        self.record_btn = tk.Button(
            frame, text="⏺ 开始录音 (⌘⇧Space)", font=("", 13, "bold"),
            height=2, command=self.toggle_recording,
            bg="#e74c3c", fg="white", relief=tk.FLAT, cursor="hand2",
        )
        self.record_btn.pack(fill=tk.X, ipady=3, pady=4)

        # 实时预览
        tk.Label(frame, text="实时识别:", anchor="w", font=("", 9)).pack(fill=tk.X)
        self.preview_label = tk.Label(frame, text="", font=("", 14),
                                      anchor="w", fg="#2c3e50", wraplength=370, justify="left")
        self.preview_label.pack(fill=tk.X, pady=(0, 6))

        # 历史文本
        tk.Label(frame, text="识别历史:", anchor="w", font=("", 9)).pack(fill=tk.X)
        self.text_area = scrolledtext.ScrolledText(frame, height=5, font=("", 12), wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # 底部按钮
        btn_row = tk.Frame(frame)
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text="📋 复制全部", command=self._copy,
                  font=("", 9), relief=tk.FLAT, bg="#3498db", fg="white").pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text="🗑️ 清空", command=self._clear,
                  font=("", 9), relief=tk.FLAT, bg="#95a5a6", fg="white").pack(side=tk.RIGHT)

        # 权限提示
        tk.Label(frame, text="流式上屏需授权: 系统设置 → 隐私与安全 → 辅助功能 → Terminal",
                 font=("", 8), fg="gray", wraplength=380).pack(fill=tk.X)

    def _setup_global_hotkey(self):
        try:
            import Quartz
            mask = Quartz.kCGEventFlagMaskCommand | Quartz.kCGEventFlagMaskShift
            def cb(proxy, etype, event, refcon):
                flags = Quartz.CGEventGetFlags(event)
                key = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventKeyCodeField)
                if (flags & mask) == mask and key == 49:  # Space
                    if etype == Quartz.kCGEventKeyDown:
                        self.root.after(0, self.toggle_recording)
                    return None
                return event
            self._event_tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly,
                Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown), cb, None,
            )
            if self._event_tap:
                src = Quartz.CFMachPortCreateRunLoopSource(None, self._event_tap, 0)
                Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
                Quartz.CGEventTapEnable(self._event_tap, True)
                print("[热键] ⌘⇧Space 已注册")
        except Exception as e:
            print(f"[热键] 不可用 ({e})")

    def toggle_recording(self):
        if not self.asr.is_recording:
            self._start()
        else:
            self._stop()

    def _start(self):
        self._typed_len = 0
        self._final_buf = ""
        self.asr.start()
        self.record_btn.config(text="⏹ 停止录音 (⌘⇧Space)", bg="#27ae60")
        self.status_label.config(text="🔴 录音中…", fg="red")
        self.preview_label.config(text="")

    def _stop(self):
        self.asr.stop()
        self.record_btn.config(text="⏺ 开始录音 (⌘⇧Space)", bg="#e74c3c")
        self.status_label.config(text="就绪", fg="gray")
        # 最后一次 flush
        time.sleep(0.3)
        self._flush_final()
        self._typed_len = 0

    def _poll_asr(self):
        """每 80ms 轮询 ASR 状态，实现流式上屏"""
        if self.asr.is_recording:
            if self.asr.partial_changed:
                self.asr.partial_changed = False
                p = self.asr.partial_text
                self.preview_label.config(text=p if p else "…")

                target = self._final_buf + p
                diff = len(target) - self._typed_len

                if diff > 0:
                    # 有新文字 → 粘贴
                    inject_text(target[self._typed_len:])
                    self._typed_len = len(target)
                elif diff < 0:
                    # partial 回退 → 删掉已上屏的，重新粘贴 final_buf
                    send_backspaces(self._typed_len)
                    self._typed_len = 0
                    if self._final_buf:
                        inject_text(self._final_buf)
                        self._typed_len = len(self._final_buf)

            if self.asr.final_changed:
                self.asr.final_changed = False
                f = self.asr.final_text
                if f:
                    self._final_buf += f
                    self.text_area.insert(tk.END, f + "\n")
                    self.text_area.see(tk.END)
                    self.preview_label.config(text="")

        self.root.after(80, self._poll_asr)

    def _flush_final(self):
        """停止录音后，确保所有文字已上屏"""
        if self.asr.final_changed:
            self.asr.final_changed = False
            f = self.asr.final_text
            if f:
                self._final_buf += f
                self.text_area.insert(tk.END, f + "\n")
                self.text_area.see(tk.END)
        # partial 中残余的文字也算 final
        if self.asr.partial_text:
            self._final_buf += self.asr.partial_text
            self.text_area.insert(tk.END, self.asr.partial_text + "\n")
            self.text_area.see(tk.END)

    def _copy(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="已复制 ✓", fg="green")
            self.root.after(2000, lambda: self.status_label.config(text="就绪", fg="gray"))

    def _clear(self):
        self.text_area.delete("1.0", tk.END)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self.asr.is_recording:
            self.asr.stop()
        if self._event_tap:
            import Quartz
            Quartz.CGEventTapEnable(self._event_tap, False)
        self.root.destroy()


# ── 入口 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="X-ASR Mac 本地语音输入法")
    ap.add_argument("--model-dir", default=os.path.join(SCRIPT_DIR, "..", "x-asr-live-demo", "models", "asr"))
    ap.add_argument("--hotwords-file", default="")
    ap.add_argument("--hotwords-score", type=float, default=1.5)
    args = ap.parse_args()

    print("[初始化] 加载模型…")
    asr = ASREngine(args.model_dir, args.hotwords_file, args.hotwords_score)
    print("[初始化] 完成\n"
          "─────────────────────────────\n"
          "  ⌘⇧Space  开始/停止录音\n"
          "  文字实时上屏到焦点应用\n"
          "─────────────────────────────")
    VoiceIMEApp(asr).run()


if __name__ == "__main__":
    main()
