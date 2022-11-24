#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
from PIL import Image
import numpy as np
import pickle
import dask.array as da
from dask import delayed
from pims import PyAVReaderIndexed
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


def extract_times(loaded_labels, loaded_times):
    frame_set = set()
    if "DLC error" in loaded_labels:
        cat_i = loaded_labels.index("DLC error")
        for ind in range(len(loaded_times)):
            for start, end, _ in loaded_times[ind][cat_i]:
                for f in range(start, end):
                    frame_set.add(f)
    return sorted(list(frame_set))


def read_video(path):
    stream = PyAVReaderIndexed(path)
    shape = stream.frame_shape
    lazy_imread = delayed(stream.get_frame)
    return (lazy_imread, shape)


def read_stack(stack, frames, shape):
    arr = np.array(
        [da.from_delayed(stack(i), shape=shape, dtype=np.uint8) for i in frames]
    )
    return arr


def extract_frames(labels_path, video_path, save_path, draw_hist):
    loaded_labels, loaded_times = load_labels(labels_path)
    if draw_hist:
        histogram(loaded_labels, loaded_times, save_path)
    if loaded_labels == 0:
        print("no frames with DLC errors")
        return [], []
    frame_list = extract_times(loaded_labels, loaded_times)
    stack, shape = read_video(video_path)
    frames = read_stack(stack, frame_list, shape)
    print(f"frames with DLC errors: {frame_list}")
    return frames, frame_list


def histogram(loaded_labels, loaded_times, save_path):
    frame_counts = defaultdict(lambda: 0)
    for label in loaded_labels:
        cat_i = loaded_labels.index(label)
        for ind in range(len(loaded_times)):
            for start, end, _ in loaded_times[ind][cat_i]:
                for f in range(start, end):
                    frame_counts[label] += 1
    nz_labels = [l for l in loaded_labels if frame_counts[l] > 0]
    plt.figure()
    plt.bar(nz_labels, [frame_counts[l] for l in nz_labels])
    plt.ylabel("number of frames")
    plt.savefig(f"{save_path}/hist.jpg")
    print("histogram ready")


@click.command()
@click.option("--video_path", required=True, help="Path to the video file")
@click.option("--labels_path", required=True, help="Path to the annotation file")
@click.option(
    "--save_path",
    required=True,
    help="The folder where the extracted frames will be saved",
)
@click.option(
    "--draw_histogram",
    is_flag=True,
    help="Draw the histogram of behaviors (saved as hist.jpg at save_path)",
)
def main(labels_path, video_path, save_path, draw_histogram):
    frames, frame_list = extract_frames(
        labels_path, video_path, save_path, draw_histogram
    )
    for frame, f_i in zip(frames, frame_list):
        im = Image.fromarray(frame)
        im.save(f"{save_path}/frame{f_i}.jpg")


if __name__ == "__main__":
    main()
