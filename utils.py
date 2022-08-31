from PyQt5.QtCore import QThread
from PyQt5.Qt import pyqtSignal
from PyQt5.QtWidgets import QWidget
import numpy as np
from collections import defaultdict
import pandas as pd
import yaml
import pickle
import dask.array as da
from tqdm import tqdm
from warnings import catch_warnings, filterwarnings
import os
import shutil
from ruamel.yaml import YAML
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

# from joblib import Parallel, delayed


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
        from dask import delayed

        stream = PyAVReaderIndexed(path)
        shape = stream.frame_shape
        lazy_imread = delayed(stream.get_frame)
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
        arr = np.array(
            [
                da.from_delayed(stack(i), shape=shape, dtype=np.uint8)
                for i in range(start, end, fs)
            ]
        )
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
            self.animals = points_df["animals"]
            self.names = points_df["names"]
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
            d = {x: self.points_df[x][animal] for x in range(start, end)}
            d["animals"] = [animal]
            d["names"] = self.names
            return PointsData(d)
        else:
            df = self.points_df.loc[list(range(start, end))]
            df = df.iloc[df.index.get_level_values(1) == animal]
            return PointsData(df)

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


def read_tracklets(filename, min_frames):
    print("loading the DLC data")
    with open(filename, "rb") as f:
        data_p = pickle.load(f)
    header = data_p["header"]
    names = header.unique("bodyparts")
    keys = sorted([key for key in data_p.keys() if key != "header"])
    coords = defaultdict(lambda: {})
    index_dict = defaultdict(lambda: [])
    animals = []
    for tr_id in tqdm(keys):
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


def read_settings(settings_file):
    with open(settings_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings

def get_settings(config_file, show_settings):
    if not os.path.exists(config_file):
        shutil.copyfile("default_config.yaml", config_file)
        show_settings = True
    else:
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
        self.set_dict()

    def load(self, file):
        with open(file, "rb") as f:
            self.array = pickle.load(f)

    def set_dict(self):
        self.count = defaultdict(lambda: 0)
        running_dict = {}
        self.boxes = []
        self.n_ind = 0
        for frame_i, frame in enumerate(self.array):
            self.boxes.append(defaultdict(lambda: [-100, -100, 10, 10, -100, -100]))
            updated = set()
            for box_i, box in enumerate(frame):
                y1, x1, y2, x2, id = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                w = np.abs(x2 - x1)
                h = np.abs(y2 - y1)
                rect_x = center_x + w / 2
                rect_y = center_y - h / 2
                if id not in running_dict.keys():
                    cur = 0
                    while cur in [running_dict[x] for x in running_dict]:
                        cur += 1
                    running_dict[id] = cur
                    if cur + 1 > self.n_ind:
                        self.n_ind = cur + 1
                updated.add(id)
                self.boxes[frame_i][running_dict[id]] = [
                    center_x,
                    center_y,
                    w,
                    h,
                    rect_x,
                    rect_y,
                ]
                self.count[id] = 0
            not_updated = set(running_dict.keys()).difference(updated)
            for id in not_updated:
                self.count[id] += 1
                if self.count[id] > self.lim_count:
                    del self.count[id]
                    del running_dict[id]

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
