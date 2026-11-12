#!/usr/bin/env python3
"""Run CellTypist-compatible inference with NumPy and a CellTypist model pickle."""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np


VERSION = "0.1.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predicted-labels", type=Path, required=True)
    parser.add_argument("--decision-matrix", type=Path, required=True)
    parser.add_argument("--probability-matrix", type=Path, required=True)
    parser.add_argument("--mode", choices=("best_match", "prob_match"), default="best_match")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--transpose-input", action="store_true")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser.parse_args()


def read_matrix(path, transpose):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(handle, dialect))

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("The expression matrix must contain identifiers, a header, and numeric values")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("The expression matrix has rows with different numbers of columns")

    column_names = np.asarray(rows[0][1:], dtype=str)
    row_names = np.asarray([row[0] for row in rows[1:]], dtype=str)
    try:
        values = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("All expression values must be numeric") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Expression values must be finite and non-negative")

    if transpose:
        return values.T, row_names, column_names
    return values, column_names, row_names


def load_model(path):
    # CellTypist model files are dictionaries containing sklearn objects.
    # Loading pickle is intentionally isolated inside the selected container.
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, dict) or not {"Model", "Scaler_"}.issubset(model):
        raise ValueError("The uploaded file is not a CellTypist model pickle")
    classifier = model["Model"]
    scaler = model["Scaler_"]
    for attribute in ("features", "classes_", "coef_", "intercept_"):
        if not hasattr(classifier, attribute):
            raise ValueError(f"CellTypist classifier is missing '{attribute}'")
    for attribute in ("mean_", "scale_"):
        if not hasattr(scaler, attribute):
            raise ValueError(f"CellTypist scaler is missing '{attribute}'")
    return classifier, scaler


def sigmoid(values):
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def predict(values, genes, classifier, scaler, mode, threshold):
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every cell must have a positive total count")
    normalized = np.log1p(values * (10000.0 / totals[:, None]))

    feature_index = {str(gene): index for index, gene in enumerate(classifier.features)}
    input_indices = []
    model_indices = []
    seen = set()
    for index, gene in enumerate(genes):
        gene = str(gene)
        if gene in feature_index and gene not in seen:
            input_indices.append(index)
            model_indices.append(feature_index[gene])
            seen.add(gene)
    if not model_indices:
        raise ValueError("No input gene symbols overlap the CellTypist model features")

    model_indices = np.asarray(model_indices, dtype=int)
    scaled = (normalized[:, input_indices] - np.asarray(scaler.mean_)[model_indices]) / np.asarray(scaler.scale_)[model_indices]
    scaled[scaled > 10] = 10
    coefficients = np.asarray(classifier.coef_)[:, model_indices]
    scores = scaled @ coefficients.T + np.asarray(classifier.intercept_)
    classes = np.asarray(classifier.classes_, dtype=str)
    if len(classes) == 2 and scores.shape[1] == 1:
        scores = np.column_stack((-scores[:, 0], scores[:, 0]))
    probabilities = sigmoid(scores)

    if mode == "best_match":
        labels = classes[np.argmax(scores, axis=1)]
    else:
        labels = np.asarray([
            "|".join(classes[row > threshold]) or "Unassigned" for row in probabilities
        ])
    return classes, scores, probabilities, labels, len(model_indices)


def write_table(path, row_names, columns, values):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(columns))
        for name, row in zip(row_names, values):
            writer.writerow([name] + list(row))


def main():
    args = parse_args()
    if not 0 <= args.probability_threshold <= 1:
        raise ValueError("Probability threshold must be between 0 and 1")
    values, genes, cells = read_matrix(args.input, args.transpose_input)
    classifier, scaler = load_model(args.model)
    classes, scores, probabilities, labels, overlap = predict(
        values, genes, classifier, scaler, args.mode, args.probability_threshold
    )
    write_table(args.predicted_labels, cells, ["predicted_labels"], labels[:, None])
    write_table(args.decision_matrix, cells, classes, scores)
    write_table(args.probability_matrix, cells, classes, probabilities)
    print(f"Predicted {len(cells)} cells using {overlap} overlapping genes", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CellTypist inference failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
