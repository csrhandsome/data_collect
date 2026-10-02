# Replay integration contract

All commands run from repository root. Python: `uv run ...`; Node: `pnpm --dir replay/frontend ...`. No ROS or Docker dependency.

Dataset root defaults to replay/demo_data; each child with meta/info.json is a dataset; child basename is dataset_id (demo_v21, demo_v30). REPLAY_DATA_ROOT can configure a directory of datasets or a single dataset. Never expose arbitrary paths via HTTP.

## Python parsing interface (replay/scripts)

- `read_dataset.py`: `read_dataset(root: Path) -> dict` with id, name, version, robot_type, fps, total_episodes, total_frames, features, episodes.
- `read_episode.py`: `read_episode(root: Path, episode_index: int) -> dict` with episode_index, length, duration_s, tasks, blocks (features list).
- `read_ee.py`: `read_ee(root: Path, episode_index: int, feature: str = "ee_pose", max_points: int = 2000) -> dict` with feature, names, units, timestamps (episode-local seconds), values (list of float lists), point_count, total_points, bounds ({min:list,max:list}). Downsampling retains first and last rows. Raise ValueError for unsupported feature, missing file, malformed data; KeyError for missing episode/feature.
- `resolve_video.py`: `resolve_video(root: Path, episode_index: int, feature: str) -> dict` with path (absolute Path), feature, start_time_s (offset within MP4), end_time_s (exclusive), duration_s, fps. v3 clips share MP4: preserve metadata offsets.
- Dataset version v2.0/v2.1 episode files + JSONL metadata; v3.0 chunk files + episodes parquet, filter episode_index even with shared files, honor metadata paths/chunk and file indices. No lerobot/torch dependency. Repository fields: ee_pose=[x,y,z,roll,pitch,yaw]; video exterior_image_1_left / wrist_image_left.
- Feature object: key, label, kind ('video'|'ee'|'unsupported'), dtype, shape:list[int], names:list[str]|null. Include ALL info.json features, unsupported non-draggable. EE recognizes ee_pose / observation.ee_pose / observation.end_effector_pose / ee_position / observation.ee_position.
- `generate_demo.py`: CLI `python -m replay.scripts.generate_demo [--output PATH]`, create demo_v21 and demo_v30 with 3 episodes each, 30Hz, 180 rows/episode; actual browser playable H264 MP4 of synthetic robot/workbench, 2 cameras. v3 combines episodes in parquet and video, nonzero offsets for ep1/ep2. Safe rerun only generated demo targets; do not delete arbitrary datasets. Include metadata features and stats. Generated files ignored, include generation instructions.
- `inspect_dataset.py`: CLI dataset path + optional episode index, emit readable JSON/summary; useful outside HTTP.

## HTTP API (replay/backend)

- GET /api/health -> {status:'ok'}
- GET /api/datasets -> {datasets:[{id,name,version,robot_type,fps,total_episodes,total_frames}]}
- GET /api/datasets/{id} -> dataset object above including features and episodes
- GET /api/datasets/{id}/episodes -> {episodes:[{episode_index,length,duration_s,tasks}]}
- GET /api/datasets/{id}/episodes/{index} -> {dataset_id,episode_index,length,duration_s,tasks,blocks}
- GET /api/datasets/{id}/episodes/{index}/ee?feature=ee_pose&max_points=2000 -> {dataset_id,episode_index,feature,names,units,timestamps,values,point_count,total_points,bounds}
- GET /api/datasets/{id}/episodes/{index}/video?feature=exterior_image_1_left -> {dataset_id,episode_index,feature,url,start_time_s,end_time_s,duration_s,fps}, url relative /api/.../video/content?feature=...
- GET /api/datasets/{id}/episodes/{index}/video/content?feature=... -> video/mp4 with byte Range support (206, Accept-Ranges, invalid range 416). HEAD optional.
- Errors missing dataset/episode/feature/file=404, malformed dataset/unsupported field=422, invalid args=422; no paths leaked.
- FastAPI app factory `create_app(data_root: Path | None = None)` for API tests. `replay.backend.main:app`. route modules under api; use case functions under services, ONE public function per service file (list_datasets/get_dataset/get_episode/get_ee/get_video/stream_video). Shared dependencies, config, response models separately.

## React frontend (replay/frontend)

React + Vite + TypeScript + Tailwind CSS + pnpm, separate pages / routes / components / hooks / lib / types. Vite proxies /api to localhost:8000 (config env override optional). Chinese labels, polished dark engineering workstation, teal/cyan accent. Left DataSidebar + reusable DataBlock shows dataset, format, episode, all fields & shapes (unsupported visible disabled). Right ReferencePanel accepts native HTML drag/drop plus accessible Add buttons. Reusable VisualizationCard hosts VideoPlayer or EETrajectory. EE includes XYZ time-series and spatial trajectory, meters, axis labels, tooltip/playhead. Video respects v3 start/end offsets (native full-file controls cannot freely cross episode boundary); custom local controls ideal. Single workspace timeline drives video & EE; remove/clear cards; dataset/episode switch resets stale cards; empty/loading/error states, no embedded mock fallback. Route `/` or `/replay` acceptable, persistent selected dataset/episode in query params helpful. Responsive layouts. Minimize dependencies. Test build and browser flow.
