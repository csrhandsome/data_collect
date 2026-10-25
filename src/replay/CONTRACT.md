# Replay integration contract

All commands run from repository root. Python: `uv run ...`; Node: `pnpm --dir src/replay/frontend ...`. No ROS or Docker dependency.

Dataset root defaults to data/dataset for both collection and replay; each direct child with meta/info.json is a dataset; child basename is dataset_id. Hub repo namespaces do not add local directory levels. REPLAY_DATA_ROOT can configure a directory of datasets or a single dataset. The frontend scans on load and has a rescan button to refresh datasets and episode metadata. Never expose arbitrary paths via HTTP.

## Python parsing interface (src/replay/scripts)

- `read_dataset.py`: `read_dataset(root: Path) -> dict` with id, name, version, robot_type, fps, total_episodes, total_frames, features, episodes.
- `read_episode.py`: `read_episode(root: Path, episode_index: int) -> dict` with episode_index, length, duration_s, tasks, blocks (features list), success (boolean/null), saved_at_ns (string/null).
- `read_ee.py`: `read_ee(root: Path, episode_index: int, feature: str = "ee_pose", max_points: int = 2000) -> dict` with feature, names, units, timestamps (episode-local seconds), values (list of float lists), point_count, total_points, bounds ({min:list,max:list}). Downsampling retains first and last rows. Raise ValueError for unsupported feature, missing file, malformed data; KeyError for missing episode/feature.
- `resolve_video.py`: `resolve_video(root: Path, episode_index: int, feature: str) -> dict` with path (absolute Path), feature, start_time_s (offset within MP4), end_time_s (exclusive), duration_s, fps. v3 clips share MP4: preserve metadata offsets.
- Dataset version v3.0 only: shared Parquet/MP4 shards and episode Parquet metadata; honor chunk/file indices, global row boundaries and per-camera video timestamps. No embedded image support.
- Feature object: key, label, kind ('video'|'audio'|'series'|'ee'|'unsupported'), dtype, shape:list[int], names:list[str]|null. Include ALL info.json features, unsupported non-draggable. EE recognizes ee_pose / observation.ee_pose / observation.end_effector_pose / ee_position / observation.ee_position.
- `generate_demo.py`: create demo_v30 with 3 episodes, shared Parquet and browser-playable H.264 MP4 cameras; preserve nonzero episode video offsets.
- `inspect_dataset.py`: CLI dataset path + optional episode index, emit readable JSON/summary; useful outside HTTP.

## HTTP API (src/replay/backend)

- GET /api/health -> {status:'ok'}
- GET /api/datasets -> {datasets:[{id,name,version,robot_type,fps,total_episodes,total_frames}]}
- GET /api/datasets/{id} -> dataset object above including features and episodes
- GET /api/datasets/{id}/episodes -> {episodes:[{episode_index,length,duration_s,tasks}]}
- GET /api/datasets/latest-episode -> {episode: {dataset_id,episode_index,saved_at_ns} | null}; only completed collection publications with saved_at_ns are considered. Timestamp is a decimal string to preserve nanosecond precision in JavaScript. This route precedes /datasets/{id}.
- GET /api/datasets/{id}/episodes/{index} -> {dataset_id,episode_index,length,duration_s,tasks,blocks,success,saved_at_ns}; missing historical annotation returns success=null.
- POST /api/datasets/{id}/episodes/{index}/annotation; JSON {success: true | false | null, expected_saved_at_ns?: string | null} -> updated EpisodeDetail. Requires an explicit boolean or null; extra fields are rejected. Optional publication timestamp detects a stale/renumbered episode (409). Merge success atomically into the existing audio/ or root-level sync sidecar, preserving other fields; create a root sidecar if absent. No Parquet changes. Serialize with task starts and reject annotation during active deletion (409). Same-origin write rules apply. Shared sidecar writes by VAD/audio-window jobs merge only their fields under the same process-safe file lock.
- GET /api/datasets/{id}/episodes/{index}/ee?feature=ee_pose&max_points=2000 -> {dataset_id,episode_index,feature,names,units,timestamps,values,point_count,total_points,bounds}
- GET /api/datasets/{id}/episodes/{index}/video?feature=exterior_image_1_left -> {dataset_id,episode_index,feature,url,start_time_s,end_time_s,duration_s,fps}, url relative /api/.../video/content?feature=...
- GET /api/datasets/{id}/episodes/{index}/video/content?feature=... -> video/mp4 with byte Range support (206, Accept-Ranges, invalid range 416). HEAD optional.
- `/video` and `/video/content` accept dtype=video only, directly serving MP4 files and respecting v3 episode timestamp offsets. No conversion/cache path.
- Legacy `/image` and `/image/metadata` endpoints, ImageStream and read_image have been removed; no per-frame image requests during playback.
- Errors missing dataset/episode/feature/file=404, malformed dataset/unsupported field=422, invalid args=422; no paths leaked.
- FastAPI app factory `create_app(data_root: Path | None = None)` for API tests. `replay.backend.main:app`. route modules under api; use case functions under services, ONE public function per service file (list_datasets/get_dataset/get_episode/get_ee/get_video/stream_video). Shared dependencies, config, response models separately.

## Local operations

- `GET /api/operations` -> `{operation: Operation | null}`; returns the active or last task. Status survives browser reload while the backend is running.
- `POST /api/operations` -> 202 `Operation`; JSON `{kind: "collect" | "replay" | "delete", dataset_id?: string, episode_index?: int}`. Extra fields, arbitrary paths and commands are rejected.
- `collect`: fixed `uv run vr_collect` in repository root; saving uses panda.yaml rather than the dataset selected in the viewer. Y saves without assigning success; X discards without assigning failure. Seal each saved episode's Parquet/metadata and publish the sync sidecar last with saved_at_ns; resume lazily for the next episode. This makes each save readable before the collection process ends, at the cost of reopening the writer per episode.
- `replay`: dataset required; resolves its highest saved episode index. If a supplied index differs, returns 409 so the user can refresh and confirm the new target. Fixed CLI `python -m replay.scripts.replay_robot`, public Panda controller only. Load and validate trajectory before connecting; prefer 100 Hz measured trace, then sync frames, then full EE/joint Parquet states. Gripper events pause playback timing. `--dry-run` uses FakeBackend.
- `delete`: dataset and episode required; v3.0 MP4, via `uv run episode-delete --yes`, using `EpisodeDeleter` as the single business API. Deletes and renumbers data, videos, audio, root-level trace/sync sidecars, episode/frame indices, splits and statistics. v3 uses the upstream dataset tools within the same transaction. Rejects symlinks/path escapes. No stop during the mutation.
- `POST /api/operations/{id}/stop` -> `Operation`; SIGINT request for collect/replay, wait for cleanup before reporting a terminal state. 404 for a stale task ID.
- Operation fields: `id`, `kind`, `state` (`running`, `stopping`, `succeeded`, `failed`, `cancelled`), `dataset_id`, `episode_index`, `started_at`, `finished_at`, `return_code`, `logs` (last 80 lines, 1500 characters/line). One active task per backend process; concurrent starts return 409. Use one backend worker. Do not run terminal collection/inference concurrently. Backend shutdown requests cooperative stop and waits up to 30 seconds; stop long tasks explicitly before shutting down.
- Browser write requests require matching Origin/Host; Vite preserves Host when proxying. Service remains local; no remote authentication is implemented.
- UI: operation confirmations, reusable `WaitingState` with elapsed time and reduced motion support, polling/stop/logs, automatic dataset refresh on task completion. Clear stale views for replay/deletion; keep collection views. Do not send hardware commands from browser tests.

## React frontend (src/replay/frontend)

React + Vite + TypeScript + Tailwind CSS + pnpm, separate pages / routes / components / hooks / lib / types. Vite proxies /api to localhost:8003 (config env override optional). Chinese labels, black and white editorial workspace. Left DataSidebar + reusable DataBlock shows dataset, format, episode, all fields & shapes (unsupported visible disabled). Right ReferencePanel accepts native HTML drag/drop plus accessible Add buttons. Reusable VisualizationCard hosts video, image, audio, numeric plots or EE. EE includes XYZ time-series and spatial trajectory, meters, axis labels, tooltip/playhead. Video respects v3 start/end offsets. Single workspace timeline drives all views; native video runs independently of timeline paints and seeks only for explicit seek revisions, playback transitions or recovery after a stall; remove/clear cards; dataset/episode switch resets stale cards; empty/loading/error states, no embedded mock fallback. Route `/` or `/replay`, selected dataset/episode in query params. Responsive layouts. Minimize dependencies. Test build and browser flow.

Poll latest-episode every 1.5 seconds, including during collection and for collections started from the terminal. The first result is a baseline so reload preserves the requested episode; later publications refresh datasets, switch to the saved dataset/episode, and add all camera views plus the first EE view. Automatic following can be unchecked. Annotation updates do not change the publication token or cause navigation. Show success/failure/unannotated buttons with aria-pressed state; save on click, disable while pending, show retryable errors and update displayed state only after the server succeeds. Navigating while a request finishes must not update another episode's UI.
