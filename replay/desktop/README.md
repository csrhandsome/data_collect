# Electron + Nuitka 桌面版

当前构建目标是 **Ubuntu 24.04 / Linux x86_64**。Electron 使用现有 React 界面，
FastAPI 在随机的 `127.0.0.1` 端口同时提供页面和 API。桌面版不使用 Vite 服务，
也不在运行时调用 `uv`、`pnpm` 或源码目录。

`runtime/desktop_runtime.py`、`control`、`replay.backend`、`replay.scripts`、
`vr_collect` 和音频 VAD 模块由 Nuitka **module 模式**编译到一个 `.so` 中。
随包提供 Python 3.12、第三方依赖和 Franka MJCF 模型；第三方库保持上游形式，
兼容 PyTorch JIT、动态加载与设备库。公开的 `bootstrap.py` 只负责路径、环境隔离和
多进程启动，不包含采集、回放或删除业务。这个版本不是 Nuitka standalone/onefile 模式。

自有 Python 业务源码、原始 TSX、真实数据集和机器人账号密码不会复制到分发包。
Electron 的 JS 和归档仍可提取，二进制也仍可逆向，打包不提供绝对的代码保密。

## 构建

需要 Python 3.12、uv、pnpm 10.32.1、Node.js、C/C++ 编译工具和 patchelf，
以及项目 Python 原生依赖所需的系统库。首次构建需要下载依赖和 Electron。

在仓库根目录运行：

```bash
bash replay/desktop/scripts/build.sh
```

输出：

```text
dist/electron/panda-data-workbench-0.1.0-x86_64.AppImage
dist/electron/linux-unpacked/panda-data-workbench
```

当前包携带锁定环境中的 PyTorch/CUDA 等依赖，AppImage 约 4.7 GiB。
此版本优先保留依赖兼容性，尚未做 CPU 专用环境或依赖裁剪。
编译从 `build/desktop/sources/` 中的固定快照进行，分发包内的
`resources/runtime/build-manifest.json` 记录该快照的文件哈希和编译时间。

开发期间只改自有 Python 代码时，可以复用已复制的第三方依赖；依赖变化后必须完整构建：

```bash
uv run --locked --group packaging python replay/desktop/scripts/build-runtime.py --reuse-dependencies
pnpm --dir replay/frontend build
pnpm --dir replay/desktop package
```

## 使用

双击 AppImage，或从终端启动：

```bash
./dist/electron/panda-data-workbench-0.1.0-x86_64.AppImage
```

本次测试主机的 Electron SUID 沙箱助手未配置，默认启动会报
`The SUID sandbox helper binary was found, but is not configured correctly`。
在这种环境下，本次验证使用以下命令（它会关闭 Chromium 进程沙箱）：

```bash
./dist/electron/panda-data-workbench-0.1.0-x86_64.AppImage --no-sandbox
```

默认启动方式需要目标机器正确支持 Electron 的 Linux 沙箱；本次没有修改系统权限。

默认用户目录为 `~/.config/panda-data-workbench/`，首次启动会创建：

- `panda.yaml`：可编辑的配置，默认模板不含机器人账号密码。
- `workspace/data/dataset/`：采集保存及回放扫描目录。
- `logs/backend.log`：后端日志。
- `cache/`：Hugging Face 等缓存。

配置里的相对数据路径以 `workspace/` 为基准。真实硬件使用前，设置设备及机器人配置，
账号密码可以通过现有的 `FRANKA_USERNAME` / `FRANKA_PASSWORD` 环境变量提供。
可以使用已有配置和数据目录：

```bash
DATA_COLLECT_CONFIG=/absolute/path/panda.yaml \
REPLAY_DATA_ROOT=/absolute/path/datasets \
./dist/electron/panda-data-workbench-0.1.0-x86_64.AppImage
```

`REPLAY_DATA_ROOT` 只改变回放扫描位置；采集保存位置由配置的 `dataset.root` 决定。
`DATA_COLLECT_USER_DATA` 可指定独立用户目录。测试时设置 `DATA_COLLECT_SIMULATE=1`
使用模拟采集和模拟机器人回放；删除操作仍会真实修改选中的数据，测试只使用临时合成数据。

关闭窗口时，应用会请求采集/回放停止并等待清理和保存；如果正在删除，会等待事务完成。
应用不会以超时为由强杀任务。后端意外退出或启动失败会显示错误并提供日志位置。

AppImage 需要可用的 FUSE；不支持 FUSE 的环境可以先用 `--appimage-extract` 解压，
再运行 `squashfs-root/AppRun`，或运行整个 `linux-unpacked/` 目录中的程序。
当前 electron-builder 附带的 AppImage runtime 不支持 `--appimage-extract-and-run`。
原生依赖及驱动仍需目标机器支持，
例如桌面图形库、PortAudio、USB 权限、RealSense 和机器人网络环境。
当前只在本机 Ubuntu 24.04 测试，不声称支持所有 Linux 发行版。

## 测试

源码回归与启动/退出测试：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
pnpm --dir replay/desktop test
```

**针对实际打包目录**的集成检查（需要桌面显示器或 Xvfb）：

```bash
pnpm --dir replay/desktop test:packaged
# 无桌面的 CI，先安装 Xvfb：
xvfb-run -a pnpm --dir replay/desktop test:packaged
# 验证 AppImage 本身（测试自动解压并运行其中的实际程序）：
DATA_COLLECT_ELECTRON_EXECUTABLE="$PWD/dist/electron/panda-data-workbench-0.1.0-x86_64.AppImage" \
pnpm --dir replay/desktop test:packaged
# 同时运行 agent-browser 界面检查：
DATA_COLLECT_BROWSER_CHECK=1 pnpm --dir replay/desktop test:packaged
```

测试在 `/tmp` 创建合成数据，使用打包运行时生成 demo，并打开打包后的 Electron。
覆盖原生库/模型资源/多进程、v2/v3 视频播放与 seek、EE 曲线、图像视频缓存、
模拟采集保存、模拟机器人回放、两种格式的数据删除与重编号、页面路由、渲染错误，
以及任务运行中关闭窗口后的后端清理。启动目录位于仓库之外，还会注入无效的
`PYTHONHOME` / `PYTHONPATH` 验证运行时隔离。

结果位于 `replay/desktop/test-results/packaged-smoke.json`，
截图为 `replay/desktop/test-results/electron-workbench.png`。
真实机器人、相机、麦克风和 VR 头显尚未在此桌面产物上验证。
自动化测试在隔离的测试显示器中为 Electron 传入 `--no-sandbox`；应用的生产配置启用
`sandbox`、`contextIsolation` 并禁用 Node 集成。生产启动需要系统支持 Electron 沙箱。

官方参考：[Nuitka module 模式](https://nuitka.net/user-documentation/use-cases.html)、
[Electron ASAR](https://github.com/electron/asar)、
[electron-builder Linux](https://www.electron.build/linux)。
