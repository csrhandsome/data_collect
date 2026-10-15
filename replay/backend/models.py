"""Validated, reusable HTTP response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class APIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Feature(APIModel):
    key: str
    label: str
    kind: Literal["video", "ee", "image", "audio", "series", "unsupported"]
    dtype: str
    shape: list[int]
    names: list[str] | None = None


class EpisodeSummary(APIModel):
    episode_index: int
    length: int
    duration_s: float
    tasks: list[str]


class DatasetSummary(APIModel):
    id: str
    name: str
    version: str
    robot_type: str | None
    fps: float
    total_episodes: int
    total_frames: int


class DatasetDetail(DatasetSummary):
    features: list[Feature]
    episodes: list[EpisodeSummary]


class DatasetList(APIModel):
    datasets: list[DatasetSummary]


class EpisodeList(APIModel):
    episodes: list[EpisodeSummary]


class EpisodeDetail(EpisodeSummary):
    dataset_id: str
    blocks: list[Feature]
    success: StrictBool | None = None
    saved_at_ns: str | None = None


class EpisodeAnnotation(APIModel):
    model_config = ConfigDict(extra="forbid")
    success: StrictBool | None
    expected_saved_at_ns: str | None = None


class LatestEpisode(APIModel):
    dataset_id: str
    episode_index: int
    saved_at_ns: str


class LatestEpisodeResponse(APIModel):
    episode: LatestEpisode | None


class Bounds(APIModel):
    min: list[float]
    max: list[float]


class EETrajectory(APIModel):
    dataset_id: str
    episode_index: int
    feature: str
    names: list[str]
    units: list[str]
    timestamps: list[float]
    values: list[list[float]]
    point_count: int
    total_points: int
    bounds: Bounds


class VideoMetadata(APIModel):
    dataset_id: str
    episode_index: int
    feature: str
    url: str
    start_time_s: float = Field(ge=0)
    end_time_s: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
