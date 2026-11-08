"""The NAS downloads share one entry and one file list."""

import json

from data_analysis.data_download.catalog import DOWNLOADS, find_download
from data_analysis.data_download.cli import list_lines, main
from data_analysis.data_download.download_tactile import EXTRA_HF, HF_DATASETS, Settings, nas_ready
from data_analysis.data_download.runner import build_jobs


def test_entry_lists_the_nine_datasets():
    names = {item.name for item in DOWNLOADS}
    assert names == {
        "haptile",
        "feelsight",
        "freetacman",
        "tactile_video_pretrain_nc",
        "vitamin",
        "wirefishing",
        "sharpa_dex",
        "egocentric_100k",
    }
    batches = {item.batch for item in DOWNLOADS if item.name == "egocentric_100k"}
    assert batches == {"original", "additional", "extra_500gb"}
    text = "\n".join(list_lines())
    for name in names:
        assert name in text
    assert "doi:10.7910/DVN/MUOJXI" in text
    assert "doi:10.7910/DVN/XOPUXY" in text


def test_huggingface_folders_match_the_transport_table():
    listed = {row[0] for row in (*HF_DATASETS, *EXTRA_HF)}
    catalog = {item.folder for item in DOWNLOADS if item.kind == "hf" and item.folder in listed}
    assert catalog == listed


def test_list_command_prints_datasets(capsys):
    main(["--list"])
    output = capsys.readouterr().out
    assert "sharpa_dex" in output
    assert "extra_500gb" in output


def test_egocentric_batch_is_required():
    try:
        find_download("egocentric_100k")
    except SystemExit as exc:
        assert "original" in str(exc)
    else:
        raise AssertionError("egocentric_100k without a batch was accepted")


def test_done_record_does_not_count_without_the_nas_file(tmp_path):
    target = tmp_path / "file.tar"
    assert nas_ready(target, 4) is False
    target.write_bytes(b"1234")
    assert nas_ready(target, 4) is True
    assert nas_ready(target, 5) is False


def test_manifest_builds_jobs_for_the_selected_batch(tmp_path):
    manifest = tmp_path / "selection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo": "builddotai/Egocentric-100K",
                "revision": "abc123",
                "files": [{"path": "factory013/part.tar", "size": 8, "sha256": "aa"}],
            }
        ),
        encoding="utf-8",
    )
    spec = find_download("egocentric_100k", "additional")
    jobs = build_jobs(spec, Settings(stage=tmp_path, nas=tmp_path, endpoint="hf-mirror"), manifest)
    assert len(jobs) == 1
    assert jobs[0].dataset == "Egocentric-100K_追加1.8TB"
    assert jobs[0].rel == "factory013/part.tar"
    assert "/resolve/abc123/factory013/part.tar" in jobs[0].url
    assert jobs[0].sha256 == "aa"
