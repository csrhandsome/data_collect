"""The datasets selected from the NAS, and where each one is downloaded from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Download:
    name: str
    folder: str
    kind: str
    detail: str
    repo: str = ""
    revision: str = "main"
    dois: tuple[tuple[str, str], ...] = ()
    file_url: str = ""
    file_name: str = ""
    file_size: int = 0
    batch: str = ""


WIREFISHING_DOIS = (
    ("doi:10.7910/DVN/MUOJXI", "part1"),
    ("doi:10.7910/DVN/XOPUXY", "part2"),
)
VITAMIN_URL = (
    "https://www.modelscope.cn/datasets/Eureka1064/FTP-1-Dataset/resolve/master/ViTaMIn/ViTaMIn.tar"
)
VITAMIN_SIZE = 93_960_335_360
EGOCENTRIC_REPO = "builddotai/Egocentric-100K"

DOWNLOADS = (
    Download("haptile", "HapTile", "hf", "HapTile2026/HapTile", repo="HapTile2026/HapTile"),
    Download("feelsight", "FeelSight", "hf", "suddhu/Feelsight", repo="suddhu/Feelsight"),
    Download("freetacman", "FreeTacMan", "hf", "aCannedFish/FreeTacMan", repo="aCannedFish/FreeTacMan"),
    Download(
        "tactile_video_pretrain_nc",
        "tactile-video-pretrain-nc",
        "hf",
        "yxma/tactile-video-pretrain-nc",
        repo="yxma/tactile-video-pretrain-nc",
    ),
    Download(
        "vitamin",
        "ViTaMIn",
        "file",
        VITAMIN_URL,
        file_url=VITAMIN_URL,
        file_name="ViTaMIn.tar",
        file_size=VITAMIN_SIZE,
    ),
    Download(
        "wirefishing",
        "WireFishing-M",
        "dataverse",
        "part1 doi:10.7910/DVN/MUOJXI; part2 doi:10.7910/DVN/XOPUXY",
        dois=WIREFISHING_DOIS,
    ),
    Download(
        "sharpa_dex",
        "SharpaDex-v1.0",
        "hf",
        "Sharpa-Robotics/SharpaDex-v1.0",
        repo="Sharpa-Robotics/SharpaDex-v1.0",
        revision="5c3715ace9a09a23fc79e7d21e75b4c06e19e859",
    ),
    Download(
        "egocentric_100k",
        "Egocentric-100K_1.8TB",
        "manifest",
        EGOCENTRIC_REPO,
        repo=EGOCENTRIC_REPO,
        batch="original",
    ),
    Download(
        "egocentric_100k",
        "Egocentric-100K_追加1.8TB",
        "manifest",
        EGOCENTRIC_REPO,
        repo=EGOCENTRIC_REPO,
        batch="additional",
    ),
    Download(
        "egocentric_100k",
        "Egocentric-100K_追加500GB",
        "manifest",
        EGOCENTRIC_REPO,
        repo=EGOCENTRIC_REPO,
        batch="extra_500gb",
    ),
)


def find_download(name: str, batch: str = "") -> Download:
    matches = [item for item in DOWNLOADS if item.name == name and item.batch == batch]
    if matches:
        return matches[0]
    batches = [item.batch for item in DOWNLOADS if item.name == name and item.batch]
    if batches:
        raise SystemExit(f"{name} 需要 --batch，可选: {', '.join(batches)}")
    known = ", ".join(sorted({item.name for item in DOWNLOADS}))
    raise SystemExit(f"未知数据集 {name}。可选: {known}")
