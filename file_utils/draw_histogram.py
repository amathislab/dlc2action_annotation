#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import os
import pickle
import click
from collections import defaultdict
from matplotlib import pyplot as plt


def load_labels(labels_file):
    try:
        with open(labels_file, "rb") as f:
            _, loaded_labels, _, loaded_times = pickle.load(f)
        return loaded_labels, loaded_times
    except:
        print("annotation file is invalid or does not exist")
        return 0, 0


def add_histogram(loaded_labels, loaded_times, frame_counts, occurences):
    for cat_i, label in enumerate(loaded_labels):
        for ind in range(len(loaded_times)):
            for start, end, _ in loaded_times[ind][cat_i]:
                if occurences:
                    frame_counts[label] += 1
                else:
                    for f in range(start, end):
                        frame_counts[label] += 1
    return frame_counts


def draw_histogram(
    labels_path,
    suffix,
    print_count=False,
    min_counts=0,
    save_path=None,
    occurences=False,
    log=True,
):
    files = [
        os.path.join(labels_path, file)
        for file in os.listdir(labels_path)
        if file.endswith(suffix)
    ]
    frame_counts = defaultdict(lambda: 0)
    for file in files:
        labels, times = load_labels(file)
        if labels != 0:
            frame_counts = add_histogram(labels, times, frame_counts, occurences)
    nz_labels = [l for l in frame_counts if frame_counts[l] > min_counts]
    nz_labels = sorted(nz_labels, key=lambda x: frame_counts[x], reverse=True)
    if print_count:
        total = 0
        for v, k in sorted([(v, k) for k, v in frame_counts.items()]):
            print(f"{k}: {v}")
            total += v
        print(f"total: {total}")
    nz_labels.remove("DLC error")
    plt.figure(figsize=(20, 10))
    if log:
        plt.yscale("log")
    plt.bar(nz_labels, [frame_counts[l] for l in nz_labels])
    plt.xticks(nz_labels, rotation="vertical")
    if occurences:
        plt.ylabel("number of occurences")
    else:
        plt.ylabel("number of frames")
    plt.tight_layout()
    if save_path is None:
        save_path = f"{labels_path}/hist.jpg"
    plt.savefig(save_path, dpi=1000)
    print("histogram ready")


@click.command()
@click.option(
    "--labels_path", required=True, help="Path to the folder with annotation files"
)
@click.option(
    "--suffix",
    help="The suffix of the annotation files",
    default=("_banty.pickle", "_banty.h5"),
)
@click.option(
    "--print_count", is_flag=True, help="Print the frame counts for all labels"
)
@click.option(
    "--min_counts", default=0, help="Minimum number of counts to be displayed"
)
@click.option(
    "--save_path",
    default=None,
    help="The path to save the histogram (labels path / hist.jpg if None)",
)
@click.option(
    "--occurences",
    is_flag=True,
    help="If true, occurence count is displayed instead of frame count",
)
@click.option("--log", is_flag=True, help="If True, use logariphmic scale")
def main(labels_path, suffix, print_count, min_counts, save_path, occurences, log):
    draw_histogram(
        labels_path, suffix, print_count, min_counts, save_path, occurences, log
    )


if __name__ == "__main__":
    main()
