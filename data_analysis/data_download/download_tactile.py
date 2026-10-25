#!/usr/bin/env python3
"""Download tactile datasets onto mirnagroup local disk, then copy them to the NAS.

Skipped on purpose: Hoi! full and Daimon-Infinity full.
WireFishing-M does not fit on the local disk, so each finished file is copied
to the NAS and removed locally before the next batch is allowed to grow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF_DATASETS = (
    ("HapTile", "HapTile2026/HapTile", 0),
    ("FeelSight", "suddhu/Feelsight", 1),
    ("FreeTacMan", "aCannedFish/FreeTacMan", 2),
)
DATAVERSE = (
    ("WireFishing-M", "doi:10.7910/DVN/MUOJXI", "part1", 4),
    ("WireFishing-M", "doi:10.7910/DVN/XOPUXY", "part2", 4),
)
DRIVE_FOLDER = "1NDasyshDCL9aaQzxjn_-Q5MBURRT360B"
PROBE_URLS = {
    "hf-mirror": "https://hf-mirror.com/datasets/suddhu/Feelsight/resolve/main/feelsight_real.tar.gz",
    "huggingface": "https://huggingface.co/datasets/suddhu/Feelsight/resolve/main/feelsight_real.tar.gz",
}
MIN_FREE = 80 * 1024**3
COPY_BUF = 8 * 1024**2
UA = "tactile-pretrain-downloader/1.0"
UV = Path.home() / ".local" / "bin" / "uv"


@dataclass
class Job:
    dataset: str
    rel: str
    url: str
    size: int
    md5: str = ""
    sha256: str = ""
    priority: int = 5


@dataclass
class Settings:
    stage: Path
    nas: Path
    endpoint: str = "hf-mirror"
    file_workers: int = 4
    connections: int = 4
    max_http: int = 8
    probe_mbps: float = 0.0
    http: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(8))


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def fetch_json(url: str, timeout: int = 180):
    last: Exception | None = None
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"列表请求失败 {attempt}/5: {exc}")
            time.sleep(min(20, 3 * attempt))
    assert last is not None
    raise last


def ensure_uv_env(stage: Path) -> None:
    """Install gdown with uv and re-execute inside that virtualenv."""
    if not UV.is_file():
        raise SystemExit(f"找不到 uv: {UV}")
    venv_python = stage / ".venv" / "bin" / "python"
    if not venv_python.exists():
        log(f"用 uv 创建环境 {stage / '.venv'}")
        subprocess.check_call([str(UV), "venv", str(stage / ".venv")])
    probe = subprocess.run([str(venv_python), "-c", "import gdown"], capture_output=True)
    if probe.returncode != 0:
        log("用 uv 安装 gdown")
        subprocess.check_call([str(UV), "pip", "install", "--python", str(venv_python), "gdown"])
    # uv 的 venv python 是指向系统解释器的符号链接，不能用 resolve 后的路径比较。
    if Path(sys.prefix).resolve() != (stage / ".venv").resolve():
        log(f"切换到 uv 环境 {venv_python}")
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


def curl(args: list[str], timeout: int) -> None:
    cmd = [
        "curl",
        "-4",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "20",
        "--retry",
        "4",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        *args,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"curl {proc.returncode}: {detail}")


def remote_size(url: str) -> int:
    header = Path("/tmp") / f"tactile-probe-{os.getpid()}.hdr"
    try:
        curl(
            ["--range", "0-0", "--max-time", "60", "-D", str(header), "-o", "/dev/null", url],
            timeout=90,
        )
        text = header.read_text(encoding="latin1", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("content-range:"):
                total = line.rsplit("/", 1)[-1].strip()
                if total.isdigit():
                    return int(total)
    finally:
        header.unlink(missing_ok=True)
    return 0


def probe(url: str, workers: int) -> float:
    size = 4 * 1024**2
    started = time.perf_counter()
    errors: list[str] = []

    def one(i: int) -> None:
        try:
            curl(
                ["--range", f"0-{size - 1}", "--max-time", "90", "-o", "/dev/null", url],
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(errors[0])
    elapsed = max(time.perf_counter() - started, 0.001)
    return workers * size / elapsed / 1e6


def choose_settings(cfg: Settings) -> None:
    best_host = ""
    best_mbps = 0.0
    for name, url in PROBE_URLS.items():
        try:
            mbps = probe(url, 4)
        except Exception as exc:  # noqa: BLE001
            log(f"测速 {name} 失败: {exc}")
            continue
        log(f"测速 {name} 4连接 {mbps:.2f} MB/s")
        if mbps > best_mbps:
            best_host, best_mbps = name, mbps
    if not best_host:
        log("两个 Hugging Face 入口都测速失败，继续用 hf-mirror")
        best_host = "hf-mirror"
    url = PROBE_URLS[best_host]
    chosen_n = 4
    chosen = best_mbps
    for workers in (8, 16):
        try:
            mbps = probe(url, workers)
        except Exception as exc:  # noqa: BLE001
            log(f"测速 {best_host} {workers}连接失败: {exc}")
            break
        log(f"测速 {best_host} {workers}连接 {mbps:.2f} MB/s")
        if mbps < chosen * 1.12:
            break
        chosen_n = workers
        chosen = mbps
    cfg.endpoint = best_host
    cfg.probe_mbps = chosen
    cfg.connections = 4 if chosen_n >= 8 else 2
    cfg.file_workers = max(4, min(8, chosen_n))
    cfg.max_http = max(8, chosen_n)
    cfg.http = threading.Semaphore(cfg.max_http)
    log(
        f"采用 {cfg.endpoint}，测得 {cfg.probe_mbps:.2f} MB/s，"
        f"同时下 {cfg.file_workers} 个文件，大文件 {cfg.connections} 连接，"
        f"总 HTTP 并发 {cfg.max_http}"
    )


def hf_api(cfg: Settings) -> str:
    if cfg.endpoint == "huggingface":
        return "https://huggingface.co"
    return "https://hf-mirror.com"


def hf_resolve(cfg: Settings, repo: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"{hf_api(cfg)}/datasets/{repo}/resolve/main/{quoted}"


def _next_link(link: str | None, api: str) -> str | None:
    if not link:
        return None
    for part in link.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">")
        if start < 0 or end < 0:
            continue
        url = part[start + 1 : end]
        for host in ("https://huggingface.co", "http://huggingface.co", "https://hf-mirror.com"):
            if url.startswith(host):
                return api + url[len(host) :]
        return url
    return None


def list_hf(cfg: Settings, repo: str) -> list[dict]:
    """List every file. The tree API pages results, and a single page is not the dataset."""
    api = hf_api(cfg)
    info = fetch_json(f"{api}/api/datasets/{repo}")
    siblings = info.get("siblings") or []
    expected_bytes = int(info.get("usedStorage") or 0)
    url = f"{api}/api/datasets/{repo}/tree/main?recursive=1&expand=1&limit=100"
    files: list[dict] = []
    seen: set[str] = set()
    pages = 0
    while url:
        pages += 1
        if pages > 500:
            raise RuntimeError(f"{repo} 文件列表分页过多")
        items = None
        page_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    items = json.loads(response.read().decode("utf-8"))
                    url = _next_link(response.headers.get("Link"), api)
                page_error = None
                break
            except Exception as exc:  # noqa: BLE001
                page_error = exc
                log(f"{repo} 列表第 {pages} 页失败 {attempt}/5: {exc}")
                time.sleep(min(30, 3 * attempt))
        if page_error is not None or items is None:
            raise page_error or RuntimeError(f"{repo} 列表第 {pages} 页失败")
        if not isinstance(items, list):
            raise RuntimeError(f"意外的目录响应: {repo}")
        for item in items:
            if item.get("type") != "file":
                continue
            path = item.get("path") or ""
            if not path or path in seen:
                continue
            seen.add(path)
            lfs = item.get("lfs") or {}
            oid = str(lfs.get("oid") or "")
            files.append(
                {
                    "path": path,
                    "size": int(lfs.get("size") or item.get("size") or 0),
                    "sha256": oid if len(oid) == 64 else "",
                }
            )
        if pages == 1 or pages % 5 == 0:
            log(f"{repo} 已读取 {len(files)} 个文件，第 {pages} 页")
    listed = sum(row["size"] for row in files)
    if siblings and len(files) < len(siblings):
        raise RuntimeError(f"{repo} 列表不完整: {len(files)} / {len(siblings)} 个文件")
    if expected_bytes and listed < int(expected_bytes * 0.9):
        raise RuntimeError(
            f"{repo} 体积不完整: {listed / 1e9:.2f} GB / 预期 {expected_bytes / 1e9:.2f} GB"
        )
    return files


def list_dataverse(doi: str) -> list[dict]:
    url = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=" + urllib.parse.quote(doi)
    payload = fetch_json(url, timeout=300)
    rows = []
    for item in payload["data"]["latestVersion"]["files"]:
        data_file = item.get("dataFile") or {}
        if item.get("restricted") or data_file.get("restricted"):
            continue
        name = data_file.get("filename") or item.get("label")
        folder = item.get("directoryLabel") or data_file.get("directoryLabel") or ""
        rel = str(Path(folder) / name) if folder else name
        rows.append(
            {
                "path": rel,
                "size": int(data_file.get("filesize") or 0),
                "md5": data_file.get("md5") or "",
                "id": data_file.get("id"),
            }
        )
    return rows


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add(f"{row['dataset']}/{row['rel']}")
    return done


def append_done(path: Path, dataset: str, rel: str, size: int, lock: threading.Lock) -> None:
    line = json.dumps({"dataset": dataset, "rel": rel, "size": size, "at": time.time()}, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def wait_for_disk(stage: Path, needed: int) -> None:
    while True:
        free = shutil.disk_usage(stage).free
        if free >= needed + MIN_FREE:
            return
        log(f"本机剩余 {free / 1e9:.1f} GB，等待传到 NAS 后再继续下")
        time.sleep(10)


def hash_file(path: Path, kind: str) -> str:
    digest = hashlib.new(kind)
    with path.open("rb") as handle:
        while True:
            block = handle.read(COPY_BUF)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_range(url: str, dest: Path, start: int, end: int, http: threading.Semaphore) -> None:
    expected = end - start + 1
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if dest.exists() and dest.stat().st_size == expected:
        return
    with http:
        curl(
            [
                "--range",
                f"{start}-{end}",
                "--max-time",
                "300",
                "--speed-time",
                "60",
                "--speed-limit",
                "10240",
                "-o",
                str(tmp),
                url,
            ],
            timeout=360,
        )
    if tmp.stat().st_size != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"分片大小不符 {dest.name}")
    os.replace(tmp, dest)


def download_single(url: str, dest: Path, http: threading.Semaphore) -> None:
    partial = Path(str(dest) + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with http:
        curl(
            [
                "-C",
                "-",
                "--max-time",
                "0",
                "--speed-time",
                "90",
                "--speed-limit",
                "10240",
                "-o",
                str(partial),
                url,
            ],
            timeout=24 * 3600,
        )
    os.replace(partial, dest)


def download_multi(url: str, dest: Path, size: int, cfg: Settings) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_dir = dest.with_suffix(dest.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = 16 * 1024**2
    spans = []
    start = 0
    index = 0
    while start < size:
        end = min(size - 1, start + chunk - 1)
        spans.append((index, start, end, part_dir / f"{index:06d}.part"))
        start = end + 1
        index += 1
    workers = min(cfg.connections, len(spans))
    with futures_of(workers) as pool:
        tasks = [
            pool.submit(download_range, url, path, begin, end, cfg.http)
            for _, begin, end, path in spans
        ]
        for task in tasks:
            task.result()
    partial = Path(str(dest) + ".partial")
    with partial.open("wb") as output:
        for _, _, _, path in spans:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, COPY_BUF)
            path.unlink()
        output.flush()
        os.fsync(output.fileno())
    part_dir.rmdir()
    os.replace(partial, dest)


def futures_of(workers: int):
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=workers)


def verify(path: Path, job: Job) -> None:
    actual = path.stat().st_size
    if job.size and actual != job.size:
        raise RuntimeError(f"大小不符 {job.rel}: {actual} != {job.size}")
    if job.md5 and hash_file(path, "md5") != job.md5.lower():
        raise RuntimeError(f"MD5 不符 {job.rel}")
    if job.sha256 and hash_file(path, "sha256") != job.sha256.lower():
        raise RuntimeError(f"SHA256 不符 {job.rel}")


def copy_to_nas(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(dest) + ".partial")
    with src.open("rb") as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, COPY_BUF)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != src.stat().st_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"传到 NAS 后大小不符 {dest}")
    os.replace(partial, dest)


def ship(cfg: Settings, local: Path, job: Job) -> None:
    target = cfg.nas / job.dataset / job.rel
    started = time.perf_counter()
    copy_to_nas(local, target)
    elapsed = max(time.perf_counter() - started, 0.001)
    log(f"已传到 NAS {job.dataset}/{job.rel}  {local.stat().st_size / elapsed / 1e6:.1f} MB/s")
    local.unlink()


def run_job(cfg: Settings, job: Job, done_path: Path, done: set[str], lock: threading.Lock) -> None:
    key = f"{job.dataset}/{job.rel}"
    local = cfg.stage / "files" / job.dataset / job.rel
    if key in done:
        local.unlink(missing_ok=True)
        return
    target = cfg.nas / job.dataset / job.rel
    try:
        if target.is_file() and (not job.size or target.stat().st_size == job.size):
            append_done(done_path, job.dataset, job.rel, target.stat().st_size, lock)
            done.add(key)
            local.unlink(missing_ok=True)
            return
    except OSError:
        pass
    needed = job.size or 256 * 1024**2
    wait_for_disk(cfg.stage, needed)
    last_error = ""
    for attempt in range(1, 4):
        try:
            if local.exists() and job.size and local.stat().st_size == job.size:
                pass
            elif job.size >= 64 * 1024**2:
                size = job.size or remote_size(job.url)
                if size >= 64 * 1024**2:
                    download_multi(job.url, local, size, cfg)
                else:
                    download_single(job.url, local, cfg.http)
            else:
                download_single(job.url, local, cfg.http)
            verify(local, job)
            ship(cfg, local, job)
            append_done(done_path, job.dataset, job.rel, job.size or target.stat().st_size, lock)
            with lock:
                done.add(key)
            log(f"完成 {key}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            log(f"重试 {attempt}/3 {key}: {last_error}")
            local.unlink(missing_ok=True)
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(last_error)


def enqueue_hf(cfg: Settings, jobs: queue.PriorityQueue, seq: list[int]) -> None:
    failed: list[str] = []
    for dataset, repo, priority in HF_DATASETS:
        log(f"读取 {dataset} 文件列表")
        try:
            rows = list_hf(cfg, repo)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{dataset}: {exc}")
            log(f"{dataset} 列表失败: {exc}")
            continue
        total = sum(row["size"] for row in rows)
        log(f"{dataset} {len(rows)} 个文件，{total / 1e9:.2f} GB")
        for row in rows:
            seq[0] += 1
            job = Job(
                dataset=dataset,
                rel=row["path"],
                url=hf_resolve(cfg, repo, row["path"]),
                size=row["size"],
                sha256=row["sha256"],
                priority=priority,
            )
            jobs.put((priority, seq[0], job))
    if failed:
        raise RuntimeError("; ".join(failed))


def enqueue_dataverse(jobs: queue.PriorityQueue, seq: list[int]) -> None:
    for dataset, doi, part, priority in DATAVERSE:
        log(f"读取 {dataset} {part} 文件列表")
        rows = list_dataverse(doi)
        total = sum(row["size"] for row in rows)
        log(f"{dataset} {part} {len(rows)} 个文件，{total / 1e9:.2f} GB")
        for row in rows:
            seq[0] += 1
            job = Job(
                dataset=dataset,
                rel=f"{part}/{row['path']}",
                url=f"https://dataverse.harvard.edu/api/access/datafile/{row['id']}",
                size=row["size"],
                md5=row["md5"],
                priority=priority,
            )
            jobs.put((priority, seq[0], job))


def download_touch_and_go(cfg: Settings, done_path: Path, done: set[str], lock: threading.Lock) -> None:
    import gdown  # type: ignore
    incoming = cfg.stage / "files" / "Touch-and-Go"
    incoming.mkdir(parents=True, exist_ok=True)
    wait_for_disk(cfg.stage, 20 * 1024**3)
    log("开始下载 Touch and Go")
    last_error = ""
    for attempt in range(1, 6):
        try:
            try:
                gdown.download_folder(
                    id=DRIVE_FOLDER,
                    output=str(incoming),
                    quiet=False,
                    remaining_ok=True,
                    resume=True,
                )
            except TypeError:
                gdown.download_folder(id=DRIVE_FOLDER, output=str(incoming), quiet=False)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            log(f"Touch and Go 第 {attempt}/5 次失败，60 秒后重试: {last_error}")
            time.sleep(60)
    else:
        raise RuntimeError(last_error)
    for path in incoming.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(incoming))
        job = Job(dataset="Touch-and-Go", rel=rel, url="", size=path.stat().st_size, priority=3)
        key = f"{job.dataset}/{job.rel}"
        if key in done:
            path.unlink(missing_ok=True)
            continue
        verify(path, job)
        ship(cfg, path, job)
        append_done(done_path, job.dataset, job.rel, job.size, lock)
        with lock:
            done.add(key)
        log(f"完成 {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="先下载到 mirnagroup 本机，再传到 NAS")
    parser.add_argument("--stage", type=Path, default=Path("/home/mirnagroup/tactile-stage"))
    parser.add_argument("--nas", type=Path, default=Path("/mnt/nas"))
    args = parser.parse_args()
    cfg = Settings(stage=args.stage, nas=args.nas)
    cfg.stage.mkdir(parents=True, exist_ok=True)
    ensure_uv_env(cfg.stage)
    if not cfg.nas.is_dir():
        raise SystemExit(f"NAS 不可用: {cfg.nas}")
    pid_path = cfg.stage / "download.pid"
    if pid_path.exists():
        old = pid_path.read_text().strip()
        if old.isdigit() and Path(f"/proc/{old}").exists():
            raise SystemExit(f"已有下载进程 {old}")
    pid_path.write_text(str(os.getpid()))
    done_path = cfg.stage / "done.jsonl"
    done = load_done(done_path)
    lock = threading.Lock()
    log(f"已完成记录 {len(done)} 条，本机剩余 {shutil.disk_usage(cfg.stage).free / 1e9:.1f} GB")
    choose_settings(cfg)
    jobs: queue.PriorityQueue = queue.PriorityQueue()
    seq = [0]
    listing_errors: list[str] = []

    def list_all() -> None:
        try:
            enqueue_hf(cfg, jobs, seq)
        except Exception as exc:  # noqa: BLE001
            listing_errors.append(f"Hugging Face: {exc}")
            log(f"Hugging Face 列表失败: {exc}")
        try:
            enqueue_dataverse(jobs, seq)
        except Exception as exc:  # noqa: BLE001
            listing_errors.append(f"Dataverse: {exc}")
            log(f"Dataverse 列表失败: {exc}")

    lister = threading.Thread(target=list_all, name="list")
    lister.start()
    drive_error: list[str] = []

    def drive() -> None:
        try:
            download_touch_and_go(cfg, done_path, done, lock)
        except Exception as exc:  # noqa: BLE001
            drive_error.append(str(exc))
            log(f"Touch and Go 失败: {exc}")

    drive_thread = threading.Thread(target=drive, name="touch-and-go")
    drive_thread.start()
    pending = 0
    pending_lock = threading.Lock()
    failures: list[str] = []

    def worker() -> None:
        nonlocal pending
        while True:
            item = jobs.get()
            if item is None:
                jobs.task_done()
                return
            _, _, job = item
            with pending_lock:
                pending += 1
            try:
                run_job(cfg, job, done_path, done, lock)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{job.dataset}/{job.rel}: {exc}")
                log(f"放弃 {job.dataset}/{job.rel}: {exc}")
            finally:
                with pending_lock:
                    pending -= 1
                jobs.task_done()

    threads = [threading.Thread(target=worker, name=f"dl-{i}") for i in range(cfg.file_workers)]
    for thread in threads:
        thread.start()
    lister.join()
    jobs.join()
    for _ in threads:
        jobs.put(None)
    for thread in threads:
        thread.join()
    drive_thread.join()
    if listing_errors or failures or drive_error:
        log(f"结束，但有失败项: 列表 {len(listing_errors)}，文件 {len(failures)}，Touch and Go {len(drive_error)}")
        raise SystemExit(1)
    log("全部数据集已传到 NAS")


if __name__ == "__main__":
    main()
