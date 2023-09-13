#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy
# is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
""" Classes and methods handling annotation data """
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np


def form_annotations_csv(
    animals: list[str],
    cat_labels: list[str],
    times: list[list[np.ndarray]],
) -> list[tuple[str, str, int, int]]:
    """
    Creates a human-readable list of annotations, sorted by individual, action and
    start with a 1-line header.

    Args:
        animals: the animal names
        cat_labels: the category labels
        times: the action times

    Returns:
        the human-readable annotations list of (individual, action, start, end)
    """
    annotation_data = []
    for animal_idx, animal in enumerate(animals):
        for cat_idx, category_label in enumerate(cat_labels):
            cat_times = times[animal_idx][cat_idx].tolist()
            for time in cat_times:
                annotation_data.append(
                    (animal, category_label, time[0], time[1])
                )

    return annotation_data


def save_annotations(
    output_path: Path,
    metadata: dict,
    animals: list[str],
    cat_labels: list[str],
    times: list[list[np.ndarray]],
    human_readable: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Saves annotation data to disk.

    Args:
        output_path: the file where the annotations should be saved
        metadata: the metadata to save
        animals: the animals
        cat_labels: the category labels
        times: the action times
        human_readable: save a human-readable version of the annotations
        overwrite: overwrite the output file if it exists

    Raises:
        IOError if overwrite=False but the output path already exists
    """
    if not overwrite and output_path.exists():
        raise IOError(
            f"Could not save annotation data {output_path} as overwrite=False but the "
            f"file already exists"
        )

    with open(output_path, "wb") as f:
        pickle.dump((metadata, cat_labels, animals, times), f)

    if human_readable:
        human_readable_path = output_path.with_suffix(".csv")
        human_readable_data = form_annotations_csv(
            animals=animals, cat_labels=cat_labels, times=times
        )
        with open(human_readable_path, "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(["animal", "action", "start", "end"])
            csv_writer.writerows(human_readable_data)
