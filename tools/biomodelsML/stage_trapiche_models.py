#!/usr/bin/env python3
"""Stage user-selected Trapiche model datasets into the expected layout."""

import argparse
import json
import shutil
from pathlib import Path


MODEL_PATHS = {
    "biome_herarchy_amended_1.0.json": "vector/1.0/biome_herarchy_amended_1.0.json",
    "biome_tags_1.0.json": "vector/1.0/biome_tags_1.0.json",
    "community2vec_model_v1.0.model": "vector/1.0/community2vec_model_v1.0.model",
    "community2vec_model_v1.0.wv.vectors.npy": "vector/1.0/community2vec_model_v1.0.wv.vectors.npy",
    "community2vec_model_vocab_v1.0.json": "vector/1.0/community2vec_model_vocab_v1.0.json",
    "taxonomy_graph_1.0.graphml": "vector/1.0/taxonomy_graph_1.0.graphml",
    "text_classifier_config_1.0.json": "text/1.0/config_1.0.json",
    "config_1.0.json": "text/1.0/config_1.0.json",
    "text_classifier_model_1.0.safetensors": "text/1.0/model_1.0.safetensors",
    "model_1.0.safetensors": "text/1.0/model_1.0.safetensors",
    "text_classifier_special_tokens_map_1.0.json": "text/1.0/special_tokens_map_1.0.json",
    "special_tokens_map_1.0.json": "text/1.0/special_tokens_map_1.0.json",
    "text_classifier_tokenizer_1.0.json": "text/1.0/tokenizer_1.0.json",
    "tokenizer_1.0.json": "text/1.0/tokenizer_1.0.json",
    "text_classifier_tokenizer_config_1.0.json": "text/1.0/tokenizer_config_1.0.json",
    "tokenizer_config_1.0.json": "text/1.0/tokenizer_config_1.0.json",
    "text_classifier_vocab_1.0.txt": "text/1.0/vocab_1.0.txt",
    "vocab_1.0.txt": "text/1.0/vocab_1.0.txt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    if not selected:
        raise ValueError("Select the Trapiche text and vector model files")
    args.output.mkdir(parents=True, exist_ok=False)
    staged = set()
    for item in selected:
        source = Path(item["path"])
        name = Path(item["name"]).name
        relative = MODEL_PATHS.get(name)
        if relative is None:
            raise ValueError(f"Unrecognized Trapiche model filename: {name}")
        if relative in staged:
            raise ValueError(f"More than one selected file maps to {relative}")
        staged.add(relative)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    missing = sorted(set(MODEL_PATHS.values()) - staged)
    if missing:
        raise ValueError("Missing required Trapiche model files: " + ", ".join(missing))


if __name__ == "__main__":
    main()
