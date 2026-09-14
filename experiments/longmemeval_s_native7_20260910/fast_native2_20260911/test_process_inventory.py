from pathlib import Path
import finish_and_stop as finalizer


def test_unrelated_process_does_not_require_accessible_cwd(tmp_path):
    process = tmp_path / '1234'
    process.mkdir()
    (process/'cmdline').write_bytes(b'syncthing\0serve\0')
    assert finalizer.owned_processes(Path('/workspace/assigned'), tmp_path) == []


def test_absolute_owned_runner_is_detected_without_cwd(tmp_path):
    process = tmp_path / '1235'
    process.mkdir()
    (process/'cmdline').write_bytes(('python3\0' + str(tmp_path / 'assigned') + '/lightmem_queue.py\0').encode())
    assert finalizer.owned_processes(tmp_path / 'assigned', tmp_path) == [1235]
