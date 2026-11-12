#!/usr/bin/env python3
"""Adapt the released verification manifest to BioStudies S-BAIR13 files."""

import json
from pathlib import Path


manifest_path = Path("/opt/trapiche-container/model-manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

# The model card is documentation and is not read during inference.
manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "text/1.0/README.md"]

# BioStudies preserves these JSON files with a final newline. The semantic
# content matches the upstream assets, but their byte sizes and hashes differ.
overrides = {
    "vector/1.0/biome_tags_1.0.json": {
        "size": 3555,
        "sha256": "894f3f5d6e2d08a329d8f059759c3afe980febf6d91d79e048d85e9ca19d5e36",
    },
    "vector/1.0/community2vec_model_vocab_v1.0.json": {
        "size": 328130,
        "sha256": "634fa697c19357a0a8d631bf801512911fed152b523916bfc3bc16a7d7440550",
    },
    "text/1.0/tokenizer_1.0.json": {
        "size": 706123,
        "sha256": "c16eb91243f70974ce602faaac3e9d8ebf76bcc8c7898e9e36cf8495ecdc7951",
    },
}
for entry in manifest["files"]:
    if entry["path"] in overrides:
        entry.update(overrides[entry["path"]])

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
