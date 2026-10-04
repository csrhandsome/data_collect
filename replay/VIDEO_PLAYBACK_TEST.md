# 视频播放改造与验证（2026-10-04）

相机从“时间轴更新 → 逐帧 HTTP → Parquet 图片 → PNG/JPEG 解码 → React 换图”改为“首次准备 H.264 MP4 → 磁盘缓存 → Range 文件传输 → 浏览器原生播放”。已有 MP4 继续直接播放，v3 保留共享视频中的片段偏移。旧 PNG 播放组件、读取脚本和逐帧接口已删除。

同时取消播放期间根据每次时间轴更新纠正视频 currentTime 的做法。定位使用显式 seekVersion；播放中不频繁 seek。静态数据栏使用 memo，避免被播放时钟重复渲染。

## 本机实测

Google Chrome / Playwright，三路 224×224 相机同时播放，每次测量约 8 秒。源数据为 20 FPS。下表的新方案呈现帧率通过 requestVideoFrameCallback 的 presentedFrames 增量统计，掉帧使用 getVideoPlaybackQuality。回调次数可能小于实际呈现帧数，报告中的 maxCallbackGapMs / p95CallbackGapMs 是回调间隔，不能直接当成视频停顿。

| 数据与条件 | 每路呈现 FPS | 解码掉帧 | 播放中 waiting / seek | 视频内容请求 | 逐帧图片请求 |
| --- | --- | --- | --- | --- | --- |
| 6_19_audio / episode 0 / MP4 | 19.77 | 0 / 0 / 0 | 均为 0 / 0 | 3 | 0 |
| 6_19_audio / episode 0 / MP4 / 请求延迟 120 ms | 19.78 | 0 / 0 / 0 | 均为 0 / 0 | 3 | 0 |
| 7_6_audio / episode 0 / MP4 | 19.78 | 0 / 0 / 0 | 均为 0 / 0 | 3 | 0 |

修改前，6_19_audio 同一 episode 的三路 PNG 换图事件均为 19.76 FPS，最大换图间隔约 97–99 ms，记录到至少 246 次逐帧请求（Resource Timing 默认缓冲区限制了完整计数）。**本机没有复现持续 PPT 级卡顿，不能据此声称改造提高了本机源视频 FPS。** 已验证的收益是消除播放期间的逐帧读取/传输/解码链路，在 120 ms 请求延迟下保持源帧率，且不再被自动 seek 打断。

真实数据首次转换每路耗时约 0.15–0.66 秒；6_19_audio 的缓存命中约 2.7–2.9 毫秒。生成 MP4 大小每路约 13 KB–664 KB；第二路相机内容变化少，文件较小。缓存命中跳过相机 Parquet 列和编码器。原始图像仍保存在 Parquet 中；MP4 是 CRF 18 的有损查看副本。

同步时间保留：6_19_audio 的第一段标称时长为 19.55 秒，实际同步时长为 19.843867 秒；7_6_audio 的第一段实际为 23.148291 秒。转换按 host 时间重采样到 20 FPS，采集间隙保留画面，避免压缩时间造成与轨迹/音频错位。

JSON 报告在忽略的 `data/replay_benchmarks/`：`png-baseline.json`、`mp4-real.json`、`mp4-real-latency120.json`、`mp4-real-7_6.json`。

## 自动验证

- 后端 `pytest replay/tests -q`：111 项通过，其中 20 项视频转换测试覆盖 H.264 实际解码、颜色/片段隔离、PNG/JPEG、路径图片、奇数尺寸、Range/HEAD、缓存命中/更新、源文件损坏/越界、同步时间、并发只转换一次、失败清理与重试。
- Chrome 端到端：12 项通过，覆盖两路嵌入图片转视频、3 秒实际播放超过 65 帧、无逐帧请求/反复 seek、播放中定位、暂停后定位、2 倍速、停止边界、准备失败重试，以及原有 MP4、v3 偏移、轨迹、布局和操作交互。机器人操作由测试模拟，没有发起硬件运动。
- 前端生产构建、端到端 TypeScript 检查、Ruff 和 diff 空白检查通过。
- agent-browser 实际检查本地页面、相机解码和截图；无 Vite 错误覆盖层。

准备与复测命令见 [README](README.md)。浏览器测试增加专用 `demo_z_images` 合成数据，原 MP4 演示数据保持原有生成接口。

统计接口依据：[MDN requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)、[MDN getVideoPlaybackQuality](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/getVideoPlaybackQuality)。编码参数依据：[FFmpeg 官方文档](https://www.ffmpeg.org/ffmpeg-all.html)。
