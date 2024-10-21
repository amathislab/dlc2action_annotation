#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import os
import pickle
import random
import shutil
import string
from collections import defaultdict
from datetime import datetime
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
import yaml
from PyQt5.Qt import pyqtSignal
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget
from ruamel.yaml import YAML
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from widgets.settings import SettingsWindow

try:
    import msgpack
    from pycocotools.mask import decode, encode
except ImportError:
    print("failed segmentation related imports")
    pass
try:
    import cv2
except:
    pass


class Segmentation:
    def __init__(self, file):
        iterator = self.make_iterator(file)
        with open("colors.txt") as f:
            colors = np.array(
                [
                    list(map(lambda x: float(x) / 255, line.split())) + [1]
                    for line in f.readlines()
                ]
            )
            colors = np.expand_dims(colors, 1)
            colors = np.expand_dims(colors, 1)
        self.masks = defaultdict(lambda: [])
        self.ids = defaultdict(lambda: [])
        self.mask_colors = defaultdict(lambda: [])
        self.cats = set()
        for i, frame in enumerate(iterator):
            for f in frame:
                self.mask_colors[i].append(colors[f["category_id"]])
                self.masks[i].append(f["segmentation"])
                self.ids[i].append(f["category_id"])
                self.cats.add(f["category_id"])
            self.mask_colors[i] = np.stack(self.mask_colors[i], axis=0)
            self.ids[i] = np.array(self.ids[i])
            self.masks[i] = np.array(self.masks[i])
        self.cats = sorted(list(self.cats))
        self.cat_id = []

    def make_iterator(self, file):
        with open(file, "rb") as f:
            yield from msgpack.Unpacker(f)

    def get_mask(self, frame_id):
        if len(self.cat_id) == 0:
            x = np.zeros((1, 1, 4))
            return x
        use_mask = np.isin(self.ids[frame_id], self.cat_id)
        colors = self.mask_colors[frame_id][use_mask]
        masks = np.stack([decode(x) for x in self.masks[frame_id][use_mask]], axis=0)
        masks = np.repeat(masks[:, :, :, np.newaxis], 4, axis=-1) * colors
        mask = np.sum(masks, axis=0)
        return mask


class SignalEmitter(QWidget):
    animal_changed = pyqtSignal()
    animal_clicked = pyqtSignal(str)
    points_changed = pyqtSignal()
    hovered = pyqtSignal(str)
    point_released = pyqtSignal()
    mode_changed = pyqtSignal(bool)


def read_video(path, backend="pyav"):
    if path is None:
        return 0, 0, 0
    if backend == "cv2":
        import cv2

        cap = cv2.VideoCapture(path)
        shape = cap.read()[1].shape
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, shape, length
    elif backend == "decord":
        from decord import VideoReader

        stream = VideoReader(path)
        return (stream, stream[0].shape, len(stream))
    elif backend == "pyav":
        from pims import PyAVReaderIndexed

        stream = PyAVReaderIndexed(path)
        shape = stream.frame_shape
        lazy_imread = stream.get_frame
        return lazy_imread, shape, len(stream)
    elif backend == "pyav_fast":
        from pims import Video

        stream = Video(path)
        shape = stream[0].shape
        length = len(stream)
        return stream, shape, length


def save_hdf(df, metadata, output_file):
    store = pd.HDFStore(output_file)
    store.put("annotation", df)
    store.get_storer("annotation").attrs.metadata = metadata
    store.close()


def read_skeleton(filename, data_type, likelihood_cutoff=0, min_length_frames=0):
    """Open track or tracklet DLC file"""
    if data_type == "dlc":
        if filename[-3:] == ".h5" or filename[-5:] == ".hdf5":
            df, index = read_hdf(filename, likelihood_cutoff)
        else:
            df, index = read_tracklets(filename, min_length_frames)
    elif data_type == "calms21":
        df, index = read_calms21(filename)
    # print(f'{df.keys()=}')
    return PointsData(df), index


def read_stack(stack, start, end, shape=None, backend="pyav", fs=1):
    if type(stack) is int:
        return None
    if backend == "decord":
        stack.seek(start)
        arr = []
        for _ in range(end - start):
            arr.append(stack.next().asnumpy())
        arr = np.array(arr)
        return arr
    elif backend == "cv2":
        stack.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        arr = []
        for _ in range(end - start):
            success, img = stack.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                arr.append(img)
        arr = np.array(arr)
        return arr
    elif backend == "pyav":
        arr = np.array([stack(i) for i in range(start, end, fs)])
        return arr
    elif backend == "pyav_fast":
        with catch_warnings():
            filterwarnings("ignore", message="VideoStream.seek is deprecated")
            arr = np.stack([stack[i] for i in range(start, end, fs)])
        return arr


class PointsData:
    def __init__(self, points_df):
        self.points_df = points_df
        self.dict_type = type(points_df) is dict
        if self.dict_type:
            self.animals = points_df.pop("animals")
            self.names = points_df.pop("names")
        else:
            self.animals = list(points_df.index.levels[1])
            self.names = list(points_df.index.levels[2])

    def __len__(self):
        if self.dict_type:
            return len(self.points_df)
        else:
            return len(self.points_df.index.levels[0])

    def get_coord(self, current, animal):
        if self.dict_type:
            return self.points_df[current][animal]
        else:
            return self.points_df.loc[current, animal].to_numpy()

    def get_range(self, start, end, animal):
        if self.dict_type:
            # print(f'{self.points_df[start].keys()=}')
            d = {
                x: {animal: self.points_df[x][animal]}
                for x in range(start, end)
                if animal in self.points_df[x]
            }
            d["animals"] = [animal]
            d["names"] = self.names
            return PointsData(d)
        else:
            df = self.points_df.loc[list(range(start, end))]
            df = df.iloc[df.index.get_level_values(1) == animal]
            return PointsData(df)

    def get_start_end(self, animal):
        if self.dict_type:
            frames = []
            for x in self.points_df:
                if animal in self.points_df[x]:
                    frames.append(int(x))
        else:
            frames = list(
                self.points_df.iloc[
                    self.points_df.index.get_level_values(1) == animal
                ].index.get_level_values(0)
            )
        if len(frames) > 0:
            start = min(frames)
            end = max(frames)
            return start, end + 1
        else:
            return None, None

    def set_coord(self, current, animal, point, coord):
        if self.dict_type:
            self.points_df[current][animal][point, :] = coord
        else:
            self.points_df.loc[current, animal].iloc[point, :] = coord


def read_hdf(filename, likelihood_cutoff=0):
    temp = pd.read_hdf(filename)
    temp = temp.droplevel("scorer", axis=1)
    if "individuals" not in temp.columns.names:
        old_idx = temp.columns.to_frame()
        old_idx.insert(0, "individuals", "ind0")
        temp.columns = pd.MultiIndex.from_frame(old_idx)
    temp.iloc[:, temp.columns.get_level_values(2) == "likelihood"] = temp.iloc[
        :, temp.columns.get_level_values(2) == "likelihood"
    ].fillna(0)
    df = temp.stack(["individuals", "bodyparts"])
    df.loc[df["likelihood"] < likelihood_cutoff, ["x", "y"]] = 0
    df = df[["x", "y"]]
    index = defaultdict(lambda: None)
    return df, index


def read_calms21(filename):
    f = np.load(filename, allow_pickle=True).item()["sequences"]
    keys = sorted(list(f.keys()))
    seq = keys[0]
    f = f[seq]["keypoints"]
    coords = defaultdict(lambda: {})
    for f_i, frame_array in enumerate(f):
        coords[f_i]["ind0"] = np.array(frame_array[0]).T
        coords[f_i]["ind1"] = np.array(frame_array[1]).T
    coords["names"] = [
        "nose",
        "left ear",
        "right ear",
        "neck",
        "left hip",
        "right hip",
        "tail",
    ]
    coords["animals"] = ["ind0", "ind1"]
    index_dict = defaultdict(lambda: ["ind0", "ind1"])
    return dict(coords), index_dict


def read_tracklets(filename, min_frames=0, verbose=True):
    if verbose:
        print("loading the DLC data")
    with open(filename, "rb") as f:
        data_p = pickle.load(f)
    header = data_p["header"]
    names = header.unique("bodyparts")
    # TODO: Need support for unique_bodyparts
    keys = sorted([key for key in data_p.keys() if isinstance(key, int)])
    coords = defaultdict(lambda: {})
    index_dict = defaultdict(lambda: [])
    animals = []
    if verbose:
        keys = tqdm(keys)
    for tr_id in keys:
        if len(data_p[tr_id]) < min_frames:
            continue
        animals.append(f"ind{tr_id}")
        for frame in data_p[tr_id]:
            fr_i = int(frame[5:])
            index_dict[fr_i].append(f"ind{tr_id}")
            coords[fr_i][f"ind{tr_id}"] = []
            for bp, name in enumerate(names):
                coords[fr_i][f"ind{tr_id}"].append(
                    np.nan_to_num(data_p[tr_id][frame][bp][:2])
                )
            coords[fr_i][f"ind{tr_id}"] = np.stack(coords[fr_i][f"ind{tr_id}"])
    coords["names"] = names
    coords["animals"] = animals
    return dict(coords), index_dict


# Reads a YAML file specified by settings_file
# Parses its content, and returns the configuration settings as a dictionary
def read_settings(settings_file: str):

    # Read the file if it exists
    with open(settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    return settings


def get_settings(config_file: str, show_settings: bool):

    with open(config_file) as f:
        config = YAML().load(f)
    with open("default_config.yaml") as f:
        default_config = YAML().load(f)
    to_remove = []
    for key, value in default_config.items():
        if key in config:
            to_remove.append(key)
    for key in to_remove:
        default_config.pop(key)
    config.update(default_config)
    with open(config_file, "w") as f:
        YAML().dump(config, f)

    if show_settings:
        SettingsWindow(config_file).exec_()

    return read_settings(config_file)


def save_settings(config: dict, config_file: str):
    """Save the configuration settings to a YAML file"""

    if os.path.exists(config_file):
        prev_config = read_settings(config_file)
    prev_config.update(config)

    with open(config_file, "w") as f:
        YAML().dump(config, f)


class WorkerThread(QThread):
    job_done = pyqtSignal(tuple)

    def __init__(
        self,
        stacks,
        shapes,
        old_videos,
        loading,
        loaded,
        max_len,
        current,
        buffer,
        backend,
    ):
        super(WorkerThread, self).__init__()
        self.stacks = stacks
        self.shapes = shapes
        self.loading = loading
        self.loaded = loaded
        self.backend = backend
        self.buffer = buffer
        self.videos = old_videos
        self.max_len = max_len
        self.current = current
        self.threadactive = True

    def do_work(self):
        start, end = self.loading
        #TODO fix multivideos 
        videos = [
            read_stack(stack, start, end, shape, self.backend)
            for stack, shape in zip(self.stacks, self.shapes)
            if self.threadactive
        ]
        left_shift = False

        if self.threadactive:
            if end == self.loaded[0]:
                self.videos = [
                    np.concatenate([new_video, old_video])
                    for new_video, old_video in zip(videos, self.videos)
                    if self.threadactive
                ]
                self.loaded = [start, self.loaded[1]]
                left_shift = True
            elif start == self.loaded[1]:
                self.videos = [
                    np.concatenate([old_video, new_video])
                    for new_video, old_video in zip(videos, self.videos)
                    if self.threadactive
                ]
                self.loaded = [self.loaded[0], end]
            else:
                self.videos = videos
                self.loaded = [start, end]

            shift = self.videos[0].shape[0] - self.max_len
            if shift > 0:
                if left_shift:
                    self.videos = [
                        video[:-shift] for video in self.videos if self.threadactive
                    ]
                    self.loaded[1] -= shift
                else:
                    self.videos = [
                        video[shift:] for video in self.videos if self.threadactive
                    ]
                    self.loaded[0] += shift
            shift = self.current - self.loaded[0] - self.buffer
            if shift > 0:
                self.videos = [
                    video[shift:] for video in self.videos if self.threadactive
                ]
                self.loaded[0] += shift

        if self.threadactive:
            self.job_done.emit((self.videos, self.loaded))

    def run(self):
        self.do_work()

    def stop(self):
        self.threadactive = False
        self.wait()


class BoxLoader:
    def __init__(self, detections):
        self.lim_count = 3
        self.load(detections)

    def load(self, file):
        with open(file, "rb") as f:
            array = pickle.load(f)
        self.n_ind = len(array)
        self.boxes = defaultdict(lambda: {})
        for ind in array:
            for frame in array[ind]:
                x1, y1, x2, y2 = array[ind][frame]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                w = np.abs(x2 - x1)
                h = np.abs(y2 - y1)
                rect_x = center_x + w / 2
                rect_y = center_y - h / 2
                self.boxes[frame][ind] = [
                    center_x,
                    center_y,
                    w,
                    h,
                    rect_x,
                    rect_y,
                ]
        del array
        frames = sorted(list(self.boxes.keys()))
        self.boxes = [self.boxes[frame] for frame in range(frames[-1])]

    def get_boxes(self):
        return self.boxes

    def get_n_ind(self):
        return self.n_ind


def read_labels(labels_file):
    """Open an annotation file to retrieve a metadata dictionary, a labels list, an animals list and the annotation intervals"""
    with open(labels_file, "rb") as f:
        metadata, loaded_labels, animals, loaded_times = pickle.load(f)
    return metadata, loaded_labels, animals, loaded_times


def read_calibration(calibration_filepath):
    cam_calibration = {}
    cam_names = ["aa", "ab", "ac", "ad", "b1", "b2"]
    for cam in cam_names:
        cam_calibration[cam] = np.load(
            os.path.join(calibration_filepath, f"camera_{cam}_calibration.npy"),
            allow_pickle=True,
        )[()]
    return cam_calibration


def project_pose(cam_name, xyz, cam_calibration):
    """Project 3D points to the screen coordinate system of the specified camera"""
    uv, _ = cv2.projectPoints(
        xyz,
        cam_calibration[cam_name]["r"],
        cam_calibration[cam_name]["t"],
        cam_calibration[cam_name]["Intrinsic"],
        cam_calibration[cam_name]["dist_coeff"],
    )
    return uv.reshape(-1, 2)


def get_2d_files(filenames, data, calibration_dir):
    cam_names = [filename.split("-")[0] for filename in filenames]
    cam_calibration = read_calibration(calibration_dir)
    res = []
    for cam in cam_names:
        data_2d = []
        for frame in data:
            data_2d.append(project_pose(cam, frame, cam_calibration))
        data_2d = np.stack(data_2d)
        res.append(data_2d)
    return res


def autolabel(classifier, X, y):
    X_train = X[y != -100]


def get_color(arr, name):
    alphabet = string.ascii_lowercase
    prompt = sum([alphabet.index(x.lower()) for x in name[:3]])
    random.seed(prompt)
    return random.choice(arr)
    # return arr[prompt % len(arr)]


def oks(y_true, y_pred, visibility):
    # You might want to set these global constant
    # outside the function scope
    KAPPA = np.array([1] * len(y_true))
    # The object scale
    # You might need a dynamic value for the object scale
    SCALE = 1.0

    # Compute the L2/Euclidean Distance
    distances = np.linalg.norm(y_pred - y_true, axis=-1)
    # Compute the exponential part of the equation
    exp_vector = np.exp(-(distances**2) / (2 * (SCALE**2) * (KAPPA**2)))
    # The numerator expression
    numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
    # The denominator expression
    denominator = np.sum(visibility.astype(bool).astype(int)) + 1e-7
    return numerator / denominator


def reassign(
    annotation_file_old,
    annotation_file_new,
    tracklets_old,
    tracklets_new,
    mapping_file=None,
):
    coords_old, _ = read_tracklets(tracklets_old, verbose=False)
    coords_new, _ = read_tracklets(tracklets_new, verbose=False)
    names_old = list(coords_old["names"])
    names_new = list(coords_new["names"])
    common_names = [x for x in names_new if x in names_old]
    common_bp_old = [names_old.index(x) for x in common_names]
    common_bp_new = [names_new.index(x) for x in common_names]
    with open(annotation_file_old, "rb") as f:
        data = list(pickle.load(f))
    mapping = None
    if mapping_file is not None:
        with open(mapping_file, "rb") as f:
            mapping = pickle.load(f)
    times_new = [[[] for i in data[1]] for _ in coords_new["animals"]]
    for ind_i_old, ind_old in enumerate(data[2]):
        for cat_i, cat_list in enumerate(data[3][ind_i_old]):
            for start, end, amb in cat_list:
                votes = defaultdict(lambda: 0)
                for frame in range(start, end):
                    pairs = []
                    if frame not in coords_new or frame not in coords_old:
                        continue
                    if ind_old not in coords_old[frame]:
                        continue
                    value_old = coords_old[frame][ind_old][common_bp_old]
                    visibility_old = (value_old != 0).sum(-1) != 0
                    for ind_new in coords_new[frame]:
                        value_new = coords_new[frame][ind_new][common_bp_new]
                        visibility = visibility_old * ((value_new != 0).sum(-1) != 0)
                        # visibility = np.expand_dims(visibility, -1)
                        oks_value = oks(value_old, value_new, visibility)
                        pairs.append((oks_value, ind_new))
                    max_i = np.argmax([x[0] for x in pairs])
                    votes[pairs[max_i][1]] += 1
                max_vote = 0
                winner = None
                for ind, vote in votes.items():
                    if vote > max_vote:
                        winner = ind
                        max_vote = vote
                if mapping is not None:
                    if winner in mapping:
                        winner = mapping[winner]
                if winner is None:
                    continue
                winner_i = coords_new["animals"].index(winner)
                times_new[winner_i][cat_i].append([start, end, amb])
    data[3] = times_new
    data[2] = coords_new["animals"]
    data[0]["skeleton_files"] = [tracklets_new]
    now = datetime.now()
    data[0]["remapped"] = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(annotation_file_new, "wb") as f:
        pickle.dump(data, f)


def write_detections(
    video_file, detections_file, target_file, video_w=None, video_h=None
):
    with open(detections_file, "rb") as f:
        detections = pickle.load(f)
    new_detections = defaultdict(lambda: {})
    for ind in detections:
        for frame in detections[ind]:
            new_detections[frame][ind] = detections[ind][frame]
    del detections
    video = cv2.VideoCapture(video_file)
    if video_w is None:
        video_w = int(video.get(3))  # float `width`
    if video_h is None:
        video_h = int(video.get(4))  # float `height`
    output = cv2.VideoWriter(
        target_file, cv2.VideoWriter_fourcc(*"MP4V"), 20, (video_w, video_h)
    )
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    with open("colors.txt") as f:
        colors = [list(map(lambda x: float(x), line.split())) for line in f.readlines()]
    animals = {}
    for count in tqdm(range(frame_count)):
        ok, image = video.read()
        image = cv2.resize(image, (video_w, video_h))
        if not ok:
            break
        for ind, value in new_detections[count].items():
            if ind not in animals:
                if ind.startswith("invisible"):
                    animals[ind] = (128, 128, 128)
                else:
                    animals[ind] = tuple(colors[len(animals) % len(colors)])
            color = animals[ind]
            x1, y1, x2, y2 = map(int, value)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                image, ind, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        output.write(image)
        count += 1
    video.release()
    output.release()


def overlap(bbox1, bbox2):
    x = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    y = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return x * y / (area1 + area2 - x * y)


def get_vis_score(coords, ind, frame, mapping):
    keys = [ind]
    for old, new in mapping.items():
        if new == ind:
            keys.append(old)
    value = 0
    for ind in keys:
        if ind not in coords[frame]:
            continue
        ind_value = np.sum(coords[frame][ind][:, 0] != 0) / coords[frame][ind].shape[0]
        if ind_value > value:
            value = ind_value
    return value


def get_visible_n(coords, ind, frames, visibility_min_score, mapping):
    visible = 0
    keys = [ind]
    for old, new in mapping.items():
        if new == ind:
            keys.append(old)
    for frame in frames:
        value = 0
        for ind in keys:
            if ind not in coords[frame]:
                continue
            if (
                np.sum(coords[frame][ind][:, 0] != 0)
                >= coords[frame][ind].shape[0] * visibility_min_score
            ):
                value = 1
                break
        visible += value
    return visible


def update_mapping(old, new, mapping):
    mapping[old] = new
    for o, n in mapping.items():
        if n == old:
            mapping[o] = new
    return mapping


def extract_detections(
    tracklet_file,
    target_file,
    margin=20,
    smooth=True,
    add_missing=True,
    min_len=30,
    overlap_thr=0.8,
    strict_min_len=10,
    lowess_frac=0.05,
    visibility_min_frac=0,
    visibility_min_score=0.25,
    keep_invisible=False,
):
    coords, _ = read_tracklets(tracklet_file, verbose=False)
    detections = defaultdict(lambda: {})
    mapping = {}
    folder = os.path.dirname(target_file)
    name = os.path.basename(target_file).split(".")[0]
    mapping_file = os.path.join(folder, name + "_mapping.pickle")
    for frame in coords:
        if frame in ["names", "animals"]:
            continue
        for ind, value in coords[frame].items():
            min_x = value[:, 0][value[:, 0] != 0].min() - margin
            min_y = value[:, 1][value[:, 1] != 0].min() - margin
            max_coords = value.max(axis=0) + margin
            detections[ind][frame] = [min_x, min_y, *max_coords]
    for ind in list(detections.keys()):
        if len(detections[ind]) < strict_min_len:
            detections.pop(ind)
            mapping = update_mapping(ind, None, mapping)
    if smooth:
        for ind in detections:
            frames = sorted(list(detections[ind].keys()))
            for i in range(4):
                arr = [detections[ind][frame][i] for frame in frames]
                arr = lowess(arr, frames, is_sorted=True, frac=lowess_frac, it=0)
                for frame, x in arr:
                    detections[ind][frame][i] = x
    if overlap_thr is not None:
        keys = list(detections.keys())
        key_i = 0
        while key_i < len(keys):
            ind = keys[key_i]
            key_i += 1
            if ind not in detections:
                continue
            other_inds = set()
            for frame in detections[ind]:
                other_inds.update(list(coords[frame].keys()))
            for other_ind in other_inds:
                if other_ind == ind or other_ind not in detections:
                    continue
                overlaps = [
                    overlap(detections[ind][frame], detections[other_ind][frame])
                    for frame in detections[ind]
                    if frame in detections[other_ind]
                ]
                if len([x for x in overlaps if x < overlap_thr]) < min(
                    3, len(overlaps) / 3
                ):
                    other_det = detections.pop(other_ind)
                    keys.append(ind)
                    for frame in other_det:
                        if frame not in detections[ind]:
                            detections[ind][frame] = other_det[frame]
                        else:
                            vis_other = get_vis_score(coords, other_ind, frame, mapping)
                            vis_this = get_vis_score(coords, ind, frame, mapping)
                            if vis_other > vis_this:
                                detections[ind][frame] = other_det[frame]
                    mapping[other_ind] = ind
                    mapping = update_mapping(other_ind, ind, mapping)
    for ind in list(detections.keys()):
        if len(detections[ind]) < min_len:
            detections.pop(ind)
            mapping[ind] = None
            mapping = update_mapping(ind, None, mapping)
    if visibility_min_score > 0 and visibility_min_frac > 0:
        for ind in list(detections.keys()):
            total = len(detections[ind])
            visible = get_visible_n(
                coords, ind, detections[ind].keys(), visibility_min_score, mapping
            )
            if visible / total < visibility_min_frac:
                invisible = detections.pop(ind)
                new = f"invisible{ind[3:]}" if keep_invisible else None
                mapping = update_mapping(ind, new, mapping)
                if keep_invisible:
                    detections[f"invisible{ind[3:]}"] = invisible
    if add_missing:
        for ind in detections:
            frames = sorted(list(detections[ind].keys()))
            for i, frame in enumerate(frames[:-1]):
                if frames[i + 1] - frame != 1:
                    next_frame = frames[i + 1]
                    next_bbox = np.array(detections[ind][next_frame])
                    this_bbox = np.array(detections[ind][frame])
                    step = (next_bbox - this_bbox) / (next_frame - frame)
                    for j in range(frame + 1, next_frame):
                        detections[ind][j] = this_bbox + step * (j - frame)
    with open(target_file, "wb") as f:
        pickle.dump(dict(detections), f)
    with open(mapping_file, "wb") as f:
        pickle.dump(mapping, f)
    return mapping_file


def reassign_folder(
    old_annotation_folder,
    new_annotation_folder,
    old_annotation_suffix,
    new_annotation_suffix,
    old_tracklet_suffix,
    new_tracklet_suffix,
    old_tracklet_folder=None,
    new_tracklet_folder=None,
    mapping_folder=None,
    mapping_suffix=None,
):
    if old_tracklet_folder is None:
        old_tracklet_folder = old_annotation_folder
    if new_tracklet_folder is None:
        new_tracklet_folder = new_annotation_folder
    old_annotation_files = [
        x
        for x in os.listdir(old_annotation_folder)
        if x.endswith(old_annotation_suffix)
    ]
    old_tracklet_files = [
        x for x in os.listdir(old_tracklet_folder) if x.endswith(old_tracklet_suffix)
    ]
    new_tracklet_files = [
        x for x in os.listdir(new_tracklet_folder) if x.endswith(new_tracklet_suffix)
    ]
    video_ids = []
    unmatched = []
    for file in old_annotation_files:
        video_id = file[: -len(old_annotation_suffix)]
        if video_id + old_tracklet_suffix not in old_tracklet_files:
            unmatched.append(file)
        elif video_id + new_tracklet_suffix not in new_tracklet_files:
            unmatched.append(file)
        else:
            video_ids.append(video_id)
    if len(unmatched) > 0:
        print("Unmatched files:")
        for file in unmatched:
            print(f"   {file}")
    for video_id in tqdm(video_ids):
        if mapping_folder is not None and mapping_suffix is not None:
            mapping_file = os.path.join(mapping_folder, video_id + mapping_suffix)
        else:
            mapping_file = None
        reassign(
            annotation_file_old=os.path.join(
                old_annotation_folder, video_id + old_annotation_suffix
            ),
            annotation_file_new=os.path.join(
                new_annotation_folder, video_id + new_annotation_suffix
            ),
            tracklets_old=os.path.join(
                old_tracklet_folder, video_id + old_tracklet_suffix
            ),
            tracklets_new=os.path.join(
                new_tracklet_folder, video_id + new_tracklet_suffix
            ),
            mapping_file=mapping_file,
        )
    print("Reassignment complete")


def apply_mapping(
    old_tracklet_file,
    new_tracklet_file,
    mapping_file,
):
    with open(old_tracklet_file, "rb") as f:
        data_p = pickle.load(f)
    with open(mapping_file, "rb") as f:
        mapping = pickle.load(f)
    for old_ind, new_ind in mapping.items():
        if old_ind.startswith("ind"):
            old_tr = int(old_ind[3:])
        else:
            old_tr = int(old_ind[len("invisible") :])
        if new_ind is not None:
            if new_ind.startswith("ind"):
                new_tr = int(new_ind[3:])
            else:
                new_tr = int(new_ind[len("invisible") :])
            for frame, value in data_p[old_tr].items():
                if frame not in data_p[new_tr]:
                    data_p[new_tr][frame] = value
        data_p.pop(old_tr)
    with open(new_tracklet_file, "wb") as f:
        pickle.dump(data_p, f)


def detect_and_remap(
    old_tracklet_folders,
    detection_folder,
    new_tracklet_folder=None,
    tracklet_suffix=None,
    margin=40,
    smooth=True,
    add_missing=True,
    min_len=30,
    overlap_thr=0.7,
    strict_min_len=5,
    lowess_frac=0.07,
    visibility_min_frac=0,
    visibility_min_score=0.25,
    keep_invisible=False,
    remap=False,
):
    if tracklet_suffix is None:
        tracklet_suffix = [".pickle"]
    files = defaultdict(lambda: [])
    for folder in old_tracklet_folders:
        for file in os.listdir(folder):
            if any([file.endswith(s) for s in tracklet_suffix]):
                files[folder].append(file)
    p_bar = tqdm(total=sum([len(v) for v in files.values()]))
    for folder, file_list in files.items():
        for file in file_list:
            target_file = file.split(".")[0] + "_det.pickle"
            mapping_file = extract_detections(
                tracklet_file=os.path.join(folder, file),
                target_file=os.path.join(detection_folder, target_file),
                margin=margin,
                smooth=smooth,
                add_missing=add_missing,
                min_len=min_len,
                overlap_thr=overlap_thr,
                strict_min_len=strict_min_len,
                lowess_frac=lowess_frac,
                visibility_min_frac=visibility_min_frac,
                visibility_min_score=visibility_min_score,
                keep_invisible=keep_invisible,
            )
            if remap:
                apply_mapping(
                    old_tracklet_file=os.path.join(folder, file),
                    new_tracklet_file=os.path.join(
                        new_tracklet_folder, file.split(".")[0] + "_remapped.pickle"
                    ),
                    mapping_file=mapping_file,
                )
            p_bar.update(1)
    p_bar.close()
