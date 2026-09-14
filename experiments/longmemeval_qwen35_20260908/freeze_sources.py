"""Freeze the local native baseline sources without touching existing results."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from materialize_inputs import write_once


NATIVE_ROOTS = [
    "utils", "benchmark", "methods/e_mem", "methods/e-mem", "methods/langmem",
    "methods/mem0/source", "methods/simplemem", "methods/lightmem", "methods/a_mem",
]
CONFIGS = [
    "hybrid_e_mem.yaml", "hybrid_simplemem.yaml", "sequential_mem0.yaml",
    "sequential_langmem.yaml", "hybrid_lightmem.yaml", "hybrid_a_mem.yaml",
]
EXCLUDED = ("upstream", "tests", "data", "datasets", "results", "logs",
            "__pycache__", ".git", ".venv", "venv", "node_modules")
EXTENSIONS = ("py", "toml", "yaml", "yml", "json", "md", "txt", "jinja", "jinja2")


def inventory(root: Path, selected: list[str]) -> list[Path]:
    """Use rg to enumerate code/config assets, excluding caches and outputs."""
    command = ["rg", "--files", "--hidden", "--no-ignore-vcs", *selected]
    for extension in EXTENSIONS:
        command.extend(["-g", f"*.{extension}"])
    for folder in EXCLUDED:
        command.extend(["-g", f"!**/{folder}/**"])
    command.extend(["-g", "!**/.venv*/**"])
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return [Path(line) for line in result.stdout.splitlines() if line]


def freeze(root: Path, target: Path, files: list[Path]) -> dict[str, str]:
    """Copy only new files, or verify byte-identical prior copies."""
    hashes = {}
    for relative in sorted(set(files)):
        source = (root / relative).resolve()
        destination = (target / relative).resolve()
        if not source.is_relative_to(root.resolve()) or not destination.is_relative_to(target.resolve()):
            raise ValueError("Snapshot path escaped its declared root")
        content = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError(f"Existing snapshot differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    baseline = workspace / "MemoryData"
    paths = inventory(baseline, NATIVE_ROOTS)
    paths += [Path("config") / filename for filename in CONFIGS]
    paths += [Path("scripts/run_six_baselines.py"), Path("requirements-six-baselines.txt")]
    memorydata = freeze(baseline, args.out / "MemoryData", paths)
    higmem_root = workspace / "HiGMem"
    higmem = freeze(higmem_root, args.out / "HiGMem", inventory(higmem_root, ["."]))
    manifest = {
        "status": "native_sources_frozen_not_service_verified",
        "source_files_sha256": memorydata, "higmem_files_sha256": higmem,
        "excluded_directories": list(EXCLUDED),
        "native_methods": ["e_mem", "simplemem", "mem0", "langmem",
                           "lightmem_direct_verbatim", "higmem", "a_mem"],
    }
    write_once(args.out / "source_snapshot.json", manifest)
    print(f"Frozen MemoryData={len(memorydata)} files; HiGMem={len(higmem)} files")


if __name__ == "__main__":
    main()