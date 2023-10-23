#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import math
import re
from optparse import OptionParser

import pandas as pd

"""This script can be used to split and downsample larger videos into smaller files that 
are loaded faster"""

length_regexp = "Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,"
re_length = re.compile(length_regexp)

import shlex
from subprocess import PIPE, Popen, check_call


def main():
    filename, split_length, ds, fps, skeleton_file = parse_options()
    if split_length <= 0:
        print("Split length can't be 0")
        raise SystemExit

    p1 = Popen(
        ["ffmpeg", "-i", filename], stdout=PIPE, stderr=PIPE, universal_newlines=True
    )
    # get p1.stderr as input
    output = Popen(
        ["grep", "Duration"], stdin=p1.stderr, stdout=PIPE, universal_newlines=True
    )
    p1.stdout.close()
    matches = re_length.search(output.stdout.read())
    if matches:
        video_length = (
            int(matches.group(1)) * 3600
            + int(matches.group(2)) * 60
            + int(matches.group(3))
        )
        print("Video length in seconds: {}".format(video_length))
    else:
        print("Can't determine video length.")
        raise SystemExit

    split_count = math.ceil(video_length / split_length)

    if split_count == 1:
        print("Video length is less than the target split length.")
        raise SystemExit

    for n in range(split_count):
        split_start = split_length * n
        pth, ext = filename.rsplit(".", 1)
        if ds > 1:
            cmd = "ffmpeg -i {} -ss {} -t {} {}-{}-tmp.{}".format(
                filename, split_start, split_length, pth, n, ext
            )
        else:
            cmd = "ffmpeg -i {} -ss {} -t {} {}-{}.{}".format(
                filename, split_start, split_length, pth, n, ext
            )
        print("About to run: {}".format(cmd))
        check_call(shlex.split(cmd), universal_newlines=True)
        if skeleton_file is not None:
            _, suffix = skeleton_file.split("DLC")
            suffix = "DLC" + suffix
            start = int(split_start * fps)
            end = int(start + split_length * fps)
            temp = pd.read_hdf(skeleton_file)
            temp_temp = temp.iloc[start:end:ds].reset_index()
            print("\n")
            print(f"SHAPE: {temp_temp.shape}")
            print("\n")
            temp_temp.to_hdf(f"{pth}-{n}{suffix}", key="coords")
        if ds > 1:
            cmd = f"ffmpeg -i {pth}-{n}-tmp.{ext} -vf select='not(mod(n\,{ds})),setpts=N/FRAME_RATE/TB' -r {fps} {pth}-{n}.{ext}"
            check_call(shlex.split(cmd), universal_newlines=True)
            cmd = f"rm {pth}-{n}-tmp.{ext}"
            check_call(shlex.split(cmd), universal_newlines=True)


def parse_options():
    parser = OptionParser()

    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        help="file to split, for example sample.avi",
        type="string",
        action="store",
    )
    parser.add_option(
        "-s",
        "--split-size",
        dest="split_size",
        help="split or chunk size in seconds, for example 10",
        type="int",
        action="store",
    )
    parser.add_option(
        "-d",
        "--downsample",
        dest="ds",
        help="use ds times less frames (speed up loading)",
        type="int",
        action="store",
    )
    parser.add_option(
        "-p",
        "--fps",
        dest="fps",
        help="fps of the input video",
        type="float",
        action="store",
    )
    parser.add_option(
        "-k",
        "--skeleton-file",
        dest="skeleton_file",
        help="the corresponding skeleton file",
        type="string",
        action="store",
    )
    (options, args) = parser.parse_args()

    if not options.skeleton_file:
        options.skeleton_file = None

    if options.filename and options.split_size and options.ds and options.fps:
        return (
            options.filename,
            options.split_size,
            options.ds,
            options.fps,
            options.skeleton_file,
        )

    else:
        parser.print_help()
        raise SystemExit


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
