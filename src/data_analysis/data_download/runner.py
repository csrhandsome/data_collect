"""Build one dataset's file list and hand it to the shared downloader."""

from __future__ import annotations

import json
from pathlib import Path

from data_analysis.data_download.catalog import Download
from data_analysis.data_download.download_tactile import (
    Job,
    Settings,
    choose_settings,
    drain_jobs,
    hf_resolve,
    list_dataverse,
    list_hf,
    log,
)


def _head_url(url: str) -> str:
    import urllib.request

    from data_analysis.data_download.download_tactile import UA

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.geturl()


def build_jobs(spec: Download, cfg: Settings, manifest: Path | None) -> list[Job]:
    if spec.kind == "hf":
        rows = list_hf(cfg, spec.repo, spec.revision)
        return [
            Job(
                dataset=spec.folder,
                rel=row["path"],
                url=hf_resolve(cfg, spec.repo, row["path"], spec.revision),
                size=row["size"],
                sha256=row["sha256"],
            )
            for row in rows
        ]
    if spec.kind == "dataverse":
        jobs: list[Job] = []
        for doi, part in spec.dois:
            log(f"读取 {spec.folder} {part} 文件列表")
            for row in list_dataverse(doi):
                jobs.append(
                    Job(
                        dataset=spec.folder,
                        rel=f"{part}/{row['path']}",
                        url=f"https://dataverse.harvard.edu/api/access/datafile/{row['id']}",
                        size=row["size"],
                        md5=row["md5"],
                    )
                )
        return jobs
    if spec.kind == "file":
        return [
            Job(
                dataset=spec.folder,
                rel=spec.file_name,
                url=_head_url(spec.file_url),
                size=spec.file_size,
            )
        ]
    if spec.kind == "manifest":
        if manifest is None:
            raise SystemExit(f"{spec.name} --batch {spec.batch} 需要 --manifest")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        repo = payload.get("repo") or spec.repo
        revision = payload.get("revision") or spec.revision
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise SystemExit(f"{manifest} 没有 files 列表")
        jobs = []
        for row in files:
            path = row.get("path")
            if not path:
                raise SystemExit(f"{manifest} 有一条文件记录缺少 path")
            jobs.append(
                Job(
                    dataset=spec.folder,
                    rel=path,
                    url=hf_resolve(cfg, repo, path, revision),
                    size=int(row.get("size") or 0),
                    sha256=str(row.get("sha256") or ""),
                )
            )
        return jobs
    raise SystemExit(f"未知下载类型 {spec.kind}")


def run_download(spec: Download, cfg: Settings, manifest: Path | None = None) -> None:
    if spec.kind in {"hf", "dataverse", "manifest"}:
        choose_settings(cfg)
    jobs = build_jobs(spec, cfg, manifest)
    log(f"{spec.name} {len(jobs)} 个文件 -> {cfg.nas / spec.folder}")
    drain_jobs(cfg, jobs)
