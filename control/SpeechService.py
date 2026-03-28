from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat,
)
import dashscope
import asyncio
import os
import sounddevice as sd
import numpy as np
import threading
import time
import queue
import json
import base64
from typing import List, Optional


# ==================== ASR 回调 ====================


class AsrCallback(RecognitionCallback):
    """语音识别回调"""

    def __init__(self, text_list: List[str], frames: List[np.ndarray], owner) -> None:
        self.temp_text = []
        self.text = text_list
        self.frames = frames
        self.owner = owner

    def on_open(self) -> None:
        print("[ASR] 连接已打开")

    def on_close(self) -> None:
        print("[ASR] 连接已关闭")
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_complete(self) -> None:
        print("[ASR] 识别完成")
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_error(self, message) -> None:
        print(
            f"[ASR] 错误 - request_id: {message.request_id}, message: {message.message}"
        )
        if self.temp_text:
            self.text.extend(self.temp_text)
            self.temp_text.clear()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if sentence is None:
            return

        if isinstance(sentence, list):
            sentence_list = sentence
        else:
            sentence_list = [sentence]

        for current in sentence_list:
            if "text" in current:
                print(f"[ASR] 识别文本: {current['text']}")
                self.temp_text.append(current["text"])
                if RecognitionResult.is_sentence_end(current):
                    print(f"[ASR] 句子结束 - request_id: {result.get_request_id()}")
                    last_text = current.get("text", "")
                    if last_text:
                        self.text.append(last_text)
                    self.temp_text.clear()
            if "emo_tag" in current:
                print(f"[ASR] 情绪标签: {current['emo_tag']}")
                self.owner.emo_tag = current["emo_tag"]

    def reset(self) -> None:
        self.temp_text.clear()


# ==================== TTS 回调 ====================


class TtsCallback(QwenTtsRealtimeCallback):
    """实时语音合成回调"""

    def __init__(self):
        self.complete_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.session_id = None
        self.is_playing = False
        self._error = None

    def on_open(self) -> None:
        print("[TTS] 连接已打开")
        self.complete_event.clear()
        self._error = None

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"[TTS] 连接已关闭 - code: {close_status_code}, msg: {close_msg}")
        self.complete_event.set()

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get("type", "")

            if event_type == "session.created":
                self.session_id = response.get("session", {}).get("id")
                print(f"[TTS] 会话已创建: {self.session_id}")

            elif event_type == "session.updated":
                print("[TTS] 会话配置已更新")

            elif event_type == "response.audio.delta":
                # 接收音频数据
                audio_b64 = response.get("delta", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    self.audio_queue.put(audio_bytes)

            elif event_type == "response.done":
                print("[TTS] 响应完成")

            elif event_type == "session.finished":
                print("[TTS] 会话结束")
                self.audio_queue.put(None)  # 结束标记
                self.complete_event.set()

            elif event_type == "error":
                self._error = response.get("error", {}).get("message", "Unknown error")
                print(f"[TTS] 错误: {self._error}")
                self.audio_queue.put(None)
                self.complete_event.set()

        except Exception as e:
            print(f"[TTS] 处理事件时出错: {e}")

    def wait_for_finished(self, timeout: float = 30.0) -> bool:
        """等待合成完成"""
        return self.complete_event.wait(timeout=timeout)

    def reset(self):
        """重置状态"""
        self.complete_event.clear()
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self._error = None


class SpeechService:
    """语音服务：语音识别 + 语音合成"""

    def __init__(self, enable_stt: bool = True):
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        self.dashscope_api_key = config["DASHSCOPE_API_KEY"]
        dashscope.api_key = self.dashscope_api_key
        self.enable_stt = enable_stt

        # 音频参数
        self.asr_sample_rate = 16000  # asr 采样率
        self.tts_sample_rate = 24000  # TTS 采样率
        self.channels = 1
        self.dtype = np.int16

        # asr 资源
        self.frames: List[np.ndarray] = []
        self.transcribed_text: List[str] = []
        self.emo_tag = None
        self.is_recording = False
        self.recording_thread = None
        self._recognition_started = False

        # ASR 回调和客户端
        self.asr_callback = AsrCallback(self.transcribed_text, self.frames, self)
        self.asr_client: Optional[Recognition] = None

        if self.enable_stt:
            self.asr_client = Recognition(
                model="paraformer-realtime-v2",
                format="pcm",
                sample_rate=self.asr_sample_rate,
                heartbeat=True,
                callback=self.asr_callback,
            )

            # 初始化 asr
            try:
                self.asr_client.start()
                self._recognition_started = True
            except Exception as e:
                print(f"[asr] 初始化失败: {e}")
                self._recognition_started = False

        # TTS 资源
        self.tts_callback = TtsCallback()
        self.tts_client: Optional[QwenTtsRealtime] = None
        self._tts_connected = False

        # TTS 配置
        self.tts_config = {
            "model": "qwen3-tts-flash-realtime",
            "voice": "Cherry",
            "language_type": "Chinese",
            "response_format": AudioFormat.PCM_24000HZ_MONO_16BIT,
            "mode": "server_commit",
            "speech_rate": 1.0,
        }

    # ==================== ASR 方法 ====================

    def start_recording(self) -> bool:
        """开始录音"""
        if not self.enable_stt:
            print("[ASR] STT 已关闭，跳过录音")
            return False

        if self.is_recording:
            return False

        if not self._recognition_started:
            try:
                if self.asr_client is None:
                    self.asr_client = Recognition(
                        model="paraformer-realtime-v2",
                        format="pcm",
                        sample_rate=self.asr_sample_rate,
                        heartbeat=True,
                        callback=self.asr_callback,
                    )
                self.asr_client.start()
                self._recognition_started = True
            except Exception as e:
                print(f"[ASR] 启动失败: {e}")
                return False

        if self.asr_callback:
            self.asr_callback.reset()

        self.is_recording = True
        self.frames.clear()
        self.transcribed_text.clear()
        self.emo_tag = None

        try:
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            return True
        except Exception as e:
            print(f"[ASR] 录音线程启动失败: {e}")
            self.is_recording = False
            return False

    def stop_recording(self) -> Optional[str]:
        """停止录音"""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None

        if self._recognition_started:
            try:
                self.asr_client.stop()
            except Exception as e:
                print(f"[asr] 停止失败: {e}")
            self._recognition_started = False

        if not self.transcribed_text:
            return None
        return "".join(self.transcribed_text)

    def _record_audio(self):
        """录音线程"""
        try:
            with sd.InputStream(
                samplerate=self.asr_sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            ) as stream:
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            print(f"[asr] 录音错误: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """音频输入回调"""
        if not self.is_recording:
            return

        chunk = indata.copy()
        self.frames.append(chunk)

        try:
            if self.asr_client is None:
                raise RuntimeError("ASR client 未初始化")
            self.asr_client.send_audio_frame(chunk.tobytes())
        except Exception as e:
            print(f"[ASR] 发送音频帧失败: {e}")
            self.is_recording = False
            raise sd.CallbackStop

    def record_and_transcribe(self, duration: float = 5.0) -> Optional[str]:
        """录制指定时长并转写"""
        if self.start_recording():
            time.sleep(duration)
            return self.stop_recording()
        return None

    # ==================== TTS 方法 ====================

    def _init_tts_client(self):
        """初始化 TTS 客户端"""
        if self.tts_client is not None:
            return

        self.tts_callback.reset()
        self.tts_client = QwenTtsRealtime(
            model=self.tts_config["model"],
            callback=self.tts_callback,
            url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        )

    def _connect_tts(self):
        """连接 TTS 服务"""
        if self._tts_connected:
            return True

        try:
            self._init_tts_client()
            self.tts_client.connect()
            self.tts_client.update_session(
                voice=self.tts_config["voice"],
                response_format=self.tts_config["response_format"],
                mode=self.tts_config["mode"],
                language_type=self.tts_config["language_type"],
                speech_rate=self.tts_config.get("speech_rate", 1.0),
            )
            self._tts_connected = True
            return True
        except Exception as e:
            print(f"[TTS] 连接失败: {e}")
            self._tts_connected = False
            return False

    def _disconnect_tts(self):
        """断开 TTS 连接"""
        if self.tts_client is not None:
            try:
                self.tts_client.close()
            except Exception as e:
                print(f"[TTS] 关闭连接失败: {e}")
            self.tts_client = None
            self._tts_connected = False

    def _speak_blocking(self, text: str) -> None:
        """阻塞式语音合成和播放"""
        # 重置 TTS 客户端（每次合成使用新连接）
        self._disconnect_tts()
        self.tts_callback.reset()

        if not self._connect_tts():
            print("[TTS] 无法连接到 TTS 服务")
            return

        try:
            # 分段发送文本（每段不超过 100 字符）
            chunk_size = 100
            text_chunks = [
                text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
            ]

            for chunk in text_chunks:
                print(f"[TTS] 发送文本: {chunk}")
                self.tts_client.append_text(chunk)
                time.sleep(0.05)  # 短暂延迟，避免发送过快

            # 通知服务端文本发送完毕
            self.tts_client.finish()

            # 播放音频
            self._play_audio_from_queue()

            # 等待合成完成
            self.tts_callback.wait_for_finished(timeout=30.0)

        except Exception as e:
            print(f"[TTS] 合成失败: {e}")
        finally:
            self._disconnect_tts()

    def _play_audio_from_queue(self):
        """从队列中播放音频"""
        try:
            with sd.OutputStream(
                samplerate=self.tts_sample_rate,
                channels=self.channels,
                dtype="int16",
            ) as stream:
                print("[TTS] 开始播放...")

                while True:
                    try:
                        # 从队列获取音频数据（带超时）
                        audio_bytes = self.tts_callback.audio_queue.get(timeout=5.0)

                        if audio_bytes is None:
                            # 结束标记
                            break

                        # 转换为 numpy 数组并播放
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                        stream.write(audio_np)

                    except queue.Empty:
                        # 超时，检查是否已完成
                        if self.tts_callback.complete_event.is_set():
                            break
                        continue

                # 等待播放完成
                time.sleep(0.5)
                print("[TTS] 播放完成")

        except Exception as e:
            print(f"[TTS] 播放错误: {e}")

    async def speak(self, text: str) -> None:
        """异步语音合成和播放"""
        await asyncio.to_thread(self._speak_blocking, text)

    def set_tts_voice(self, voice: str):
        """设置 TTS 音色"""
        self.tts_config["voice"] = voice

    def set_tts_speed(self, speed: float):
        """设置 TTS 语速 (0.5 - 2.0)"""
        self.tts_config["speech_rate"] = max(0.5, min(2.0, speed))

    # ==================== 属性和清理 ====================

    @property
    def text(self) -> Optional[str]:
        """获取识别的文本"""
        if not self.transcribed_text:
            return None
        return "".join(self.transcribed_text)

    @property
    def emotion(self) -> Optional[str]:
        """获取情绪标签"""
        return self.emo_tag

    def clear_text(self):
        """清除识别的文本"""
        self.transcribed_text.clear()

    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_recording()

        if self._recognition_started:
            try:
                self.asr_client.stop()
            except Exception as e:
                print(f"[ASR] 清理时停止失败: {e}")
            self._recognition_started = False

        self._disconnect_tts()
        self.frames.clear()


# ==================== 测试入口 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("SpeechService 测试")
    print("=" * 50)

    service = SpeechService()

    try:
        # 测试 TTS
        print("\n[测试] 语音合成...")
        asyncio.run(service.speak("你好，我是语音助手，很高兴为您服务！"))

        # 测试 ASR
        print("\n[测试] 语音识别...")
        print("请说话，系统将持续识别您的语音（按 Ctrl+C 退出）")

        while True:
            if service.start_recording():
                while service.is_recording:
                    time.sleep(0.1)
                    if service.text:
                        transcribed_text = service.text
                        print(f"\n识别结果: {transcribed_text}")
                        service.clear_text()

                        # 回声测试
                        if transcribed_text:
                            print("播放识别的文本...")
                            asyncio.run(service.speak(transcribed_text))
                            service.stop_recording()
                            break
            else:
                print("录音启动失败")
                break

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        service.cleanup()
        print("测试完成")
