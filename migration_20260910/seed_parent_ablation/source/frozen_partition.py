"""Select exact precommitted partition IDs without altering any sample content."""
import hashlib
import json
from pathlib import Path
from typing import Any

from transfer_data import deterministic_subset

CANONICAL_DATA_SHA256 = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
CANONICAL_DATA_BYTES = 277383467
CANONICAL_POPULATION = 500
FROZEN_MANIFEST_SHA256 = 'e3dca5fd623ace2f10158da8ef92b5a27abd8ac9b9459e300608addbb9f31fd3'
PARTITIONS = ('dev', 'validation', 'final_test')


def validate_options(dataset: str, limit: int | None, split_manifest: Path | None,
                     partition: str | None) -> None:
    if (split_manifest is None) != (partition is None):
        raise ValueError('--split-manifest and --partition must be supplied together')
    if partition is not None:
        if partition not in PARTITIONS:
            raise ValueError('Unknown frozen partition')
        if dataset != 'longmemeval':
            raise ValueError('Frozen partitions require the canonical LongMemEval-S dataset')
        if limit is not None:
            raise ValueError('--limit cannot be combined with a frozen partition')


def load_frozen_manifest(data: Path, split_manifest: Path) -> dict:
    raw = split_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError('Frozen split manifest SHA256 mismatch')
    if data.stat().st_size != CANONICAL_DATA_BYTES:
        raise ValueError('Canonical full500 dataset byte count mismatch')
    with data.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != CANONICAL_DATA_SHA256:
            raise ValueError('Canonical full500 dataset SHA256 mismatch')
    manifest = json.loads(raw)
    if (manifest['dataset']['sha256'] != CANONICAL_DATA_SHA256
            or manifest['question_count'] != CANONICAL_POPULATION):
        raise ValueError('Frozen manifest canonical population binding mismatch')
    return manifest


def unique_ids(values: Any, label: str) -> list[str]:
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(label + ' must contain unique nonempty string IDs')
    return values


def select_from_manifest(samples: list[dict], manifest: dict, partition: str) -> tuple[list[dict], dict]:
    """Generic metadata-only validation; production calls first enforce file pins.

    Base IDs are used solely to reject cross-partition contamination. Selection
    and returned objects always retain the full question ID, including _abs.
    """
    if partition not in PARTITIONS or set(manifest['partitions']) != set(PARTITIONS):
        raise ValueError('Manifest must declare dev, validation and final_test')
    ordered, selection = deterministic_subset(samples, len(samples))
    population = set(selection['selected_ids'])
    if manifest['question_count'] != len(population):
        raise ValueError('Manifest question count differs from canonical population')
    seen, cluster_owner = set(), {}
    for name in PARTITIONS:
        part = manifest['partitions'][name]
        ids = unique_ids(part['question_ids'], name)
        clusters = {qid.removesuffix('_abs') for qid in ids}
        declared_clusters = unique_ids(part['cluster_ids'], name + ' clusters')
        if (part['question_count'] != len(ids) or part['cluster_count'] != len(clusters)
                or set(declared_clusters) != clusters
                or part['abstention_question_count'] != sum(qid.endswith('_abs') for qid in ids)):
            raise ValueError('Partition question/base/abstention metadata mismatch: ' + name)
        if seen.intersection(ids) or not set(ids) <= population:
            raise ValueError('Partition IDs overlap or contain unknown IDs')
        for cluster in clusters:
            if cluster in cluster_owner:
                raise ValueError('Base/_abs cluster crosses frozen partitions: ' + cluster)
            cluster_owner[cluster] = name
        seen.update(ids)
    if seen != population or manifest['cluster_count'] != len(cluster_owner):
        raise ValueError('Partitions do not cover the full question/base population exactly')
    wanted = set(manifest['partitions'][partition]['question_ids'])
    selected = [sample for sample in ordered if sample['sample_id'] in wanted]
    selection.update(selection='frozen-partition-filtered-seeded-id-order-v1',
                     partition=partition, selected_ids=[sample['sample_id'] for sample in selected],
                     selected_count=len(selected), canonical_population_count=len(population))
    return selected, selection


def select_samples(samples: list[dict], *, dataset: str, data: Path, limit: int | None,
                   split_manifest: Path | None, partition: str | None) -> tuple[list[dict], dict]:
    validate_options(dataset, limit, split_manifest, partition)
    if partition is None:
        # Keep the inherited no-partition selection behavior exactly.
        return deterministic_subset(samples, limit or len(samples))
    manifest = load_frozen_manifest(data, split_manifest)
    selected, selection = select_from_manifest(samples, manifest, partition)
    if selection['population_count'] != CANONICAL_POPULATION:
        raise ValueError('Partition execution requires the complete canonical 500 input')
    selection.update(split_manifest_sha256=FROZEN_MANIFEST_SHA256,
                     canonical_dataset_sha256=CANONICAL_DATA_SHA256)
    return selected, selection
