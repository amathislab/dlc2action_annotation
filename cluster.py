import copy

from sklearn import datasets, decomposition, manifold
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QRadioButton,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QStackedLayout,
    QCheckBox,
    QFileDialog,
)
from PyQt5.QtCore import QTimer, QSize, QRectF, QPoint, QPointF
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.Qt import Qt, pyqtSignal
from vispy.scene import SceneCanvas
import numpy as np
import sys
import click
import pickle
import pyqtgraph as pg
import inspect
import torch
import os
from utils import read_video, read_stack, read_skeleton, get_settings
from widgets.viewbox import VideoViewBox
from widgets.dialog import EpisodeSelector, EpisodeParamsSelector, SuggestionParamsSelector
import annotator
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from typing import Iterable
from utils import get_color


class Annotation():
    def __init__(self, filename):
        self.inds = None
        self.cats = None
        self.data = None
        if filename is not None and os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    _, self.cats, self.inds, self.data = pickle.load(f)
            except pickle.UnpicklingError:
                pass

    def generate_array(self):
        length = 0
        for ind_list in self.data:
            for cat_list in ind_list:
                length = max(length, max([x[1] for x in cat_list]))
        to_remove = []
        for i, x in enumerate(self.inds):
            ok = any([len(cat_list) > 0 for cat_list in self.data[i]])
            if not ok:
                to_remove.append(i)
        inds = [x for i, x in enumerate(self.inds) if i not in to_remove]
        data = [x for i, x in enumerate(self.data) if i not in to_remove]
        for i, x in enumerate(self.cats):
            ok = any([len(ind_list[i]) > 0 for ind_list in self.data])
            if x.startswith("unknown") or not ok:
                to_remove.append(i)
        cats = [x for i, x in enumerate(self.cats) if i not in to_remove]
        for j in range(len(data)):
            data[j] = [x for i, x in enumerate(data[j]) if i not in to_remove]
        arr = -100 * np.ones(length * len(inds), len(cats))
        for ind_i, ind_list in enumerate(data):
            for cat_i, cat_list in enumerate(ind_list):
                for start, end, _ in cat_list:
                    if cats[cat_i].startswith("negative"):
                        arr[ind_i * length + start, ind_i * length + end, cat_i] = 0
                    else:
                        arr[ind_i * length + start, ind_i * length + end, cat_i] = 1
        return arr, cats


    @staticmethod
    def get_filename(filepaths, filenames, video_id, suffix):
        for fn, fp in zip(filenames, filepaths):
            if fn.split('.')[0] == video_id:
                return os.path.join(fp, video_id + suffix + '.pickle')

    def main_labels(self, start, end, ind):
        if self.data is None or ind not in self.inds:
            return "unknown"
        cats = defaultdict(lambda: 0)
        for cat, cat_list in zip(self.cats, self.data[self.inds.index(ind)]):
            for s, e, _ in cat_list:
                if s < end and e > start:
                    cats[cat] += (min(e, end) - max(s, start))
        total_annotated = sum(cats.values())
        if total_annotated == 0:
            return "unknown"
        result = set()
        for cat, ann in cats.items():
            if ann >= (end - start) * 0.1:
                if cat.startswith('negative'):
                    result.add("no behavior")
                else:
                    result.add(cat)
        if len(result) > 1:
            result = [x for x in result if x != "no behavior"]
        if len(result) == 0:
            result = ["assorted"]
        return " + ".join(sorted(list(result)))

    # @staticmethod
    # def get_training_data(filepaths, filenames, suffix):
    #     y = 0
    #     data = []
    #     cats = []
    #     for fp, fn in zip(filepaths, filenames):
    #         filename = os.path.join(fp, fn.split('.')[0] + suffix + '.pickle')
    #         arr, cat_list = Annotation(filename).generate_array()
    #         data.append(arr)
    #         cats.append(cat_list)
    #     cats_unique = list(set(cats))
    #
    #     for cat, arr in zip(cats, data):


class PlotWidget(pg.PlotWidget):
    clicked = pyqtSignal(QPointF)
    dragged = pyqtSignal(list)
    released = pyqtSignal()

    def __init__(self, select_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = False
        self.start = None
        self.add = False
        self.select_mode = select_mode

    def mousePressEvent(self, ev):
        if self.select_mode:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                self.add = True
            else:
                self.add = False
            p = self.plotItem.vb.mapSceneToView(QPoint(ev.x(), ev.y()))
            self.clicked.emit(p)
            self.pressed = True
            self.start = [p.x(), p.y()]
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.select_mode and self.pressed:
            p = self.plotItem.vb.mapSceneToView(QPoint(ev.x(), ev.y()))
            self.dragged.emit([p.x(), p.y()] + self.start)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.select_mode:
            self.pressed = False
            self.released.emit()
        else:
            super().mouseReleaseEvent(ev)


class RectItem(pg.GraphicsObject):
    def __init__(self, position, parent=None):
        super().__init__(parent)
        self.position = position

    def paint(self, painter, option, widget=None):
        col = QColor("blue")
        col.setAlphaF(0.1)
        painter.setBrush(col)
        pen = QPen(Qt.blue, 0.001)
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        painter.drawRect(QRectF(*self.position))

    def boundingRect(self):
        return QRectF(*self.position)


class VideoWindow(QWidget):
    closed = pyqtSignal(int)

    def __init__(self, window, stack, shape, points_df, frames, n, color, label):
        super(VideoWindow, self).__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.label = QLabel(label)
        self.canvas = VideoCanvas(window, stack, shape, points_df, frames, "yellow")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.canvas.native)
        self.n = n
        self.canvas.vb.border_color = color
        self.canvas.vb._border_width = 4
        self.border_color = 255 * color
        self.move(0, 0)

    def sizeHint(self):
        return QSize(400, 300)

    def closeEvent(self, a0):
        self.canvas.set_play(False)
        self.closed.emit(self.n)
        a0.accept()

    def color(self):
        return list(map(int, self.border_color))


class VideoCanvas(SceneCanvas):
    def __init__(self, window, stack, shape, points_df, frames, color):
        super(VideoCanvas, self).__init__()
        self.unfreeze()
        self.window = window
        self.play = True
        self.speed = 40
        _, start, end, ind = frames
        self.current = start
        self.length = end - start
        self.ind = ind
        self.skeleton_color = lambda x: color
        self.start = start
        video = read_stack(stack, start, end, shape)
        vb = VideoViewBox(
            n_ind=1,
            boxes=None,
            window=window,
            animals=ind,
            points_df=points_df,
            loaded=[start, end],
            current=start,
            current_animal=ind,
            video=video,
            node_transform=self.scene.node_transform,
            displayed_animals=[ind],
            skeleton_color=self.skeleton_color,
            al_mode=False,
            al_animal=None,
            correct_mode=False,
            segmentation=None,
        )
        vb.initialize(self.current, 0)
        self.central_widget.add_widget(vb)
        self.vb = vb
        self.timer = QTimer()
        self.timer.timeout.connect(self.next)
        self.set_play(True)

    def on_key_press(self, event):
        if event.key.name == "Space":
            self.set_play()
        elif event.key.name == "Right":
            self.next()
        elif event.key.name == "Left":
            self.prev()
        else:
            print(f"canvas didn't recognise key {event.key.name}")

    def next(self):
        self.set_current_frame(self.current + 1)

    def prev(self):
        self.set_current_frame(self.current - 1)

    def set_play(self, value=None):
        if value in [True, False]:
            self.play = value
        else:
            self.play = not self.play
        if self.play:
            self.timer.start(self.speed)
        else:
            self.timer.stop()

    def set_current_frame(self, value):
        next = (value - self.start) % self.length + self.start
        self.current = next
        self.vb.set_image(self.current, [self.ind], self.skeleton_color, False, None)
        self.update()


class LineEdit(QLineEdit):
    enter = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == 16777220:
            self.enter.emit()
        else:
            super(LineEdit, self).keyPressEvent(event)


class ModWindow(QWidget):
    def __init__(self, window):
        super(ModWindow, self).__init__()
        self.window = window
        if self.window.method == "raw":
            self.parameters = []
            self.args = []
        else:
            self.parameters = inspect.signature(window.dim_red_func).parameters
            self.args = list(self.parameters.keys())
            for a in ["n_components", "verbose", "n_jobs"]:
                if a in self.args:
                    self.args.remove(a)

        self.input_widget = QWidget()
        self.input = QFormLayout()
        self.input_widget.setLayout(self.input)
        self.boxes = []
        for par in self.args:
            le = LineEdit()
            le.enter.connect(self.plot_graph)
            self.boxes.append(le)
            if par in self.window.args[self.window.method]:
                value = str(self.window.args[self.window.method][par])
            else:
                value = str(self.parameters[par].default)
            le.setText(value)
            self.input.addRow(par, le)

        self.buttons = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_graph)
        self.reset_button = QPushButton("Reset to default")
        self.reset_button.clicked.connect(self.reset_args)
        self.buttons.addWidget(self.reset_button)
        self.buttons.addWidget(self.plot_button)

        urlLink = f"<a href={self.window.doc_link}>documentation</a>"
        self.label1 = QLabel(f"Modify {self.window.method} parameters")
        self.label2 = QLabel(f"Here is the {urlLink}")
        self.label2.setTextFormat(Qt.RichText)
        self.label2.setOpenExternalLinks(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.input_widget)
        self.layout.addLayout(self.buttons)

        self.setLayout(self.layout)
        self.move(0, 0)

    def reset_args(self):
        self.window.reset_args()
        self.close()

    def stop(self):
        self.close()

    def plot_graph(self):
        values = [box.text() for box in self.boxes]
        corr_values = []
        for v in values:
            try:
                corr_values.append(int(v))
            except:
                try:
                    corr_values.append(float(v))
                except:
                    corr_values.append(v)
        arg_dict = {par: v for par, v in zip(self.args, corr_values) if v != "None"}
        arg_dict["n_components"] = 2
        try:
            self.window.get_clustering(arg_dict)
            self.window.args[self.window.method] = arg_dict
        except ValueError as e:
            self.value_error(e)

    def value_error(self, e):
        print(str(e))
        msg = QMessageBox()
        msg.setWindowTitle("Value error")
        msg.setText(str(e) + "\n \n" + "Stop modifying?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.buttonClicked.connect(self.error_buttons)
        msg.exec_()

    def error_buttons(self, button):
        if button.text()[1:] == "Yes":
            self.stop()


class Console(QWidget):
    def __init__(self, window):
        super(Console, self).__init__()
        self.window = window

        self.method_label = QLabel("Dimensionality reduction method:")
        self.method_wheel = QComboBox()
        self.method_wheel.addItems(self.window.methods)
        self.method_wheel.currentTextChanged.connect(self.window.change_method)

        self.pars_button = QPushButton("Modify parameters")
        self.pars_button.clicked.connect(self.window.modify_pars)
        if self.window.method == "raw":
            self.pars_button.setEnabled(False)

        self.method_checkbox = QCheckBox("Multiple videos open")
        self.method_checkbox.toggled.connect(self.window.set_method)
        self.close_videos_button = QPushButton("Close all videos")
        self.close_videos_button.clicked.connect(self.window.close_all_videos)
        self.select_radio = QRadioButton("Select")
        self.move_radio = QRadioButton("Move")
        self.select_radio.clicked.connect(self.window.set_select_mode)
        self.move_radio.clicked.connect(self.window.set_move_mode)
        self.move_radio.setChecked(True)
        self.save_button = QPushButton("Save selection")
        self.save_button.clicked.connect(self.window.save)
        self.save_button.setDisabled(True)
        self.open_button = QPushButton("Open intervals")
        self.open_button.clicked.connect(self.window.open_intervals)
        self.open_button.setDisabled(True)
        self.auto_button = QPushButton("Autolabel")
        self.auto_button.clicked.connect(self.window.autolabel)

        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.method_checkbox)
        self.radio_layout.addWidget(self.close_videos_button)
        self.radio_layout.addSpacing(100)
        self.radio_layout.addWidget(self.select_radio)
        self.radio_layout.addWidget(self.move_radio)
        self.radio_layout.addSpacing(100)
        self.radio_layout.addWidget(self.save_button)
        self.radio_layout.addWidget(self.open_button)
        self.radio_layout.addWidget(self.auto_button)

        self.method_layout = QHBoxLayout()
        self.method_layout.addWidget(self.method_label)
        self.method_layout.addWidget(self.method_wheel)
        self.method_layout.addWidget(self.pars_button)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.method_layout)
        self.layout.addLayout(self.radio_layout)
        self.setLayout(self.layout)


class MainWindow(QWidget):

    switch = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, filenames,
        filepaths,
        open_settings,
        config_file,
        features,
        skeleton_folder,
        annotation_folder,
        feature_folder,
        feature_suffix,
        dlc2action_name,
        dlc2action_path,
        skip_dlc2action,
        *args,
        **kwargs
        ):
        super(MainWindow, self).__init__(*args, **kwargs)

        if annotation_folder is None:
            annotation_folder = feature_folder

        self.settings_file = config_file
        self.settings = get_settings(self.settings_file, open_settings)
        self.filepaths = filepaths
        self.filenames = filenames
        self.feature_files = features
        self.annotation_files = []
        self.annotation_folder = annotation_folder
        self.feature_suffix = feature_suffix
        self.feature_folder = feature_folder
        if annotation_folder is not None and not os.path.exists(annotation_folder):
            os.mkdir(annotation_folder)
        for fn, fp in zip(self.filenames, self.filepaths):
            ann_folder = annotation_folder if annotation_folder is not None else fp
            ann_path = os.path.join(ann_folder, fn.split('.')[0] + self.settings["suffix"])
            self.annotation_files.append(ann_path)
        self.skeleton_files = []
        for fn, fp in zip(self.filenames, self.filepaths):
            sk_folder = skeleton_folder if skeleton_folder is not None else fp
            sk_path = None
            for s in self.settings["DLC_suffix"]:
                path = os.path.join(sk_folder, fn.split('.')[0] + s)
                if os.path.exists(path):
                    sk_path = path
                    break
            self.skeleton_files.append(sk_path)
        self.parameters = [
            filenames,
            filepaths,
            False,
            config_file,
            features,
            skeleton_folder,
            annotation_folder,
            feature_folder,
            feature_suffix,
            dlc2action_name,
            dlc2action_path,
            skip_dlc2action,
        ]
        with open("colors.txt") as f:
            self.colors = (
                np.array([list(map(int, line.split())) for line in f.readlines()]) / 255
            )
        self.video_mode = "same"
        self.open_videos = {}
        self.color_i = 0
        self.setWindowTitle("Clustering")
        self.method = "raw"
        self.methods = ["raw", "t-SNE", "PCA", "LLE", "ICA", "FA"]
        self.rect_position = [0, 0, 0, 0]
        self.chosen = []
        self.visited = set()
        self.select_mode = False

        self.load_files()

        self.graph = PlotWidget(self.select_mode)
        self.graph.setBackground("w")
        self.graph.dragged.connect(self.set_rect)
        self.graph.released.connect(self.on_release)

        self.args = {x: {"n_components": 2} for x in self.methods}
        self.get_data()  # TODO: self.X, self.y (load from .npy), self.frames (al_points style)
        self.get_clustering(self.args[self.method])
        # self.stack, self.shape, self.len = read_video('/Users/liza/data/CAM2_2021_06_25_083528.mp4')
        # short_len = (self.len - 100) // 100
        # self.frames = [(i, i+100) for i in range(0, self.len, short_len)][:100]
        # self.skeleton_file = None
        self.dlc2action_name = dlc2action_name
        self.dlc2action_path = dlc2action_path
        self.initialize_dlc2action_project(skip_dlc2action)

        self.console = Console(self)
        self.video_layout = QHBoxLayout()
        self.video_layout.addWidget(self.console)

        self.main_layout = QStackedLayout()
        self.main_layout.setStackingMode(QStackedLayout.StackAll)
        self.main_layout.addWidget(self.graph)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.video_layout)
        self.layout.addLayout(self.main_layout)
        self.setLayout(self.layout)

    def get_color(self, name):
        return get_color(self.colors, name)

    def initialize_dlc2action_project(self, skip):
        if skip:
            self.dlc2action_name = None
            return
        from dlc2action.project import Project
        if self.dlc2action_name is None:
            i = 0
            while not Project.project_name_available(self.dlc2action_path, f'cluster_project_{i}'):
                i += 1
            self.dlc2action_name = f'cluster_project_{i}'
        project = Project(
            self.dlc2action_name,
            data_type="features",
            annotation_type="dlc",
            projects_path=self.dlc2action_path,
            data_path=self.feature_folder,
            annotation_path=self.annotation_folder,
        )
        project.update_parameters(
            {
                "data": {
                    "annotation_suffix": self.settings["suffix"],
                    "feature_suffix": self.feature_suffix,
                    "filter_annotated": True,
                    "filter_background": True,
                    "behaviors": None
                },
                "general": {
                    "metric_functions": {"f1"},
                    "only_load_annotated": True,
                    "len_segment": 128,
                    "overlap": 1/2,
                    "model_name": "mlp",
                    "exclusive": False,
                },
                "training": {
                    "test_frac": 0,
                    "val_frac": 0,
                    "to_ram": False,
                    "num_epochs": 100,
                    "lr": 1e-4,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "partition_method": "random",
                },
                "metrics": {
                    "f1": {"average": "none"},
                },
            }
        )

    def load_animals(self, skeleton_file):
        if skeleton_file is not None:
            try:
                points_df, _ = read_skeleton(skeleton_file, data_type=self.settings["data_type"])
            except:
                print("skeleton file is invalid or does not exist")
                points_df = None
        else:
            points_df = None
        return points_df

    def load_skeleton(self, skeleton_file):
        points_df = self.load_animals(skeleton_file)
        return points_df

    def load_files(self):
        self.loading_dict = {fn.split('.')[0]: (fp, fn, skeleton) for fp, fn, skeleton in zip(self.filepaths, self.filenames, self.skeleton_files)}
        self.points_df_dict = {}
        self.stacks = {}
        self.shapes = {}

    def open_video(self, name):
        fp, fn, skeleton = self.loading_dict[name]
        name = fn.split(".")[0]
        stack, shape, _ = read_video(os.path.join(fp, fn))
        self.stacks[name] = stack
        self.shapes[name] = shape
        self.points_df_dict[name] = self.load_skeleton(skeleton)

    def video_closed(self, n):
        self.open_videos.pop(n)
        self.plot_clusters()

    def clicked(self, point, ev):
        data_list = list(self.scatter.data)
        points_list = [x[9] for x in data_list]
        n = points_list.index(ev[0])
        self.visited.update([n])
        if self.video_mode == "same":
            self.close_all_videos()
        vid, start, end, ind = self.frames[n]
        label = self.labels[n]
        if label in self.label_dict:
            label = self.label_dict[label]
        if vid not in self.stacks:
            self.open_video(vid)
        if self.points_df_dict[vid] is not None:
            points_df = self.points_df_dict[vid].get_range(start, end, ind)
        else:
            points_df = None
        vw = VideoWindow(
            self,
            self.stacks[vid],
            self.shapes[vid],
            points_df,
            self.frames[n],
            n,
            self.colors[self.color_i],
            label,
        )
        vw.closed.connect(self.video_closed)
        self.color_i = (self.color_i + 1) % len(self.colors)
        vw.show()
        self.open_videos[n] = vw
        self.plot_clusters()

    def modify_pars(self):
        self.mod = ModWindow(self)
        self.mod.show()

    def get_data(self):
        if sum(x is not None for x in self.feature_files) == 0:
            import random

            n = 100
            X, y = datasets.make_blobs(
                n_samples=n, cluster_std=[1.0, 2.5, 0.5], n_features=5
            )
            self.label_dict = {0: "zero", 1: "one", 2: "two"}
            self.data = X
            # self.labels = y
            self.labels = []

            videos = [f.split(".")[0] for f in self.filenames]
            videos_rand = [videos[random.randint(0, len(videos) - 1)] for _ in range(n)]
            start_frames = list(range(0, 1000, 50))
            self.frames = [(v, start_frames[i % len(start_frames)], start_frames[i % len(start_frames)] + 50, "ind0")
                           for i, v in enumerate(videos_rand)]
            for video in videos:
                annotation = Annotation(
                    Annotation.get_filename(
                        self.filepaths,
                        self.filenames,
                        video,
                        self.settings["suffix"]
                    )
                )
                for v, s, e, clip in self.frames:
                    if v != video:
                        continue
                    main_labels = annotation.main_labels(s, e, clip)
                    self.labels.append(main_labels)
            labels = list(set(self.labels))
            self.label_dict = {i: x for i, x in enumerate(labels)}
            inv_dict = {x: i for i, x in enumerate(labels)}
            self.labels = [inv_dict[x] for x in self.labels]
        else:
            self.filenames = [x for i, x in enumerate(self.filenames) if self.feature_files[i] is not None]
            self.filepaths = [x for i, x in enumerate(self.filepaths) if self.feature_files[i] is not None]
            self.feature_files = [x for x in self.feature_files if x is not None]
            self.frames = []
            self.data = []
            self.labels = []
            frames_step = 128
            for file_i, file in enumerate(self.feature_files):
                if file.endswith("npy"):
                    video_dict = np.load(file, allow_pickle=True).item()
                else:
                    with open(file, "rb") as f:
                        video_dict = pickle.load(f)
                video = self.filenames[file_i]
                annotation = Annotation(self.annotation_files[file_i])
                min_frames = video_dict.pop("min_frames")
                clips = list(video_dict.keys())
                for clip in clips:
                    if clip in ["max_frames", "video_tag"]:
                        continue
                    clip_arr = video_dict.pop(clip)
                    n = clip_arr.shape[0]
                    if not ((n & (n-1) == 0) and n != 0):
                        clip_arr = clip_arr.T
                    for s in range(0, clip_arr.shape[-1], frames_step):
                        end = min(clip_arr.shape[-1], s + frames_step)
                        main_labels = annotation.main_labels(s, end, clip)
                        self.frames.append((video.split('.')[0], s + min_frames[clip], end + min_frames[clip], clip))
                        self.data.append(clip_arr[:, s: end].mean(-1))
                        self.labels.append(main_labels)
            labels = sorted(list(set(self.labels)))
            self.label_dict = {i: x for i, x in enumerate(labels)}
            inv_dict = {x: i for i, x in enumerate(labels)}
            self.labels = [inv_dict[x] for x in self.labels]
            if isinstance(self.data[0], np.ndarray):
                self.data = np.stack(self.data, 0)
            else:
                self.data = torch.stack(self.data, 0).numpy()

    def reset_args(self):
        self.args[self.method] = {"n_components": 2}
        self.get_clustering(self.args[self.method])

    def get_clustering(self, args):
        method_dict = {
            "PCA": decomposition.PCA,
            "t-SNE": manifold.TSNE,
            "LLE": manifold.LocallyLinearEmbedding,
            "FA": decomposition.FactorAnalysis,
            "ICA": decomposition.FastICA,
        }
        link_dict = {
            "PCA": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
            "t-SNE": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
            "LLE": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html",
            "FA": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html",
            "ICA": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA",
        }
        if self.method == "raw":
            self.components = self.data[:, :2]
            self.doc_link = ""
        else:
            self.dim_red_func = method_dict[self.method]
            self.doc_link = link_dict[self.method]

            self.dim_red = self.dim_red_func(**args)
            self.components = self.dim_red.fit_transform(self.data)

        self.plot_clusters()

    def make_brush(self, x, n):
        if x == -100 or self.label_dict[x] == "unknown":
            col = QColor("gray")
            col.setAlphaF(0.2)
        elif self.label_dict[x] == "no behavior":
            col = pg.mkColor(110, 110, 110)
        else:
            col = pg.mkColor(self.get_color(self.label_dict[x]) * 255)
        if n in self.chosen and x != -100:
            if col == Qt.blue:
                col.setAlphaF(0.1)
            else:
                col.setAlphaF(0.7)
        return QBrush(col)

    def make_pen(self, n):
        if n in self.chosen:
            return QPen(Qt.blue, 0.3)
        else:
            if n in self.visited:
                col = pg.mkColor(121,92,178)
                return QPen(col, 0.2)
            else:
                return QPen(Qt.gray, 0.2)

    def plot_clusters(self):
        self.graph.clear()
        brush = [self.make_brush(int(x), n) for n, x in enumerate(self.labels)]
        pen = [self.make_pen(n) for n in range(len(self.components))]
        self.scatter = pg.ScatterPlotItem(
            pos=self.components,
            hoverable=True,
            hoverBrush=QBrush(Qt.gray),
            brush=brush,
            symbol="o",
            pen=pen,
        )
        brush = []
        for n in self.open_videos:
            brush.append(QBrush(QColor(*self.open_videos[n].color())))
        self.scatter.addPoints(
            pos=self.components[list(self.open_videos.keys())],
            hoverable=False,
            hoverBrush=QBrush(Qt.gray),
            brush=brush,
            symbol="x",
            size=23,
            pen=QPen(Qt.white, 0.1),
        )
        self.graph.addItem(self.scatter)
        self.rect = RectItem(self.rect_position)
        self.graph.addItem(self.rect)
        self.scatter.sigClicked.connect(self.clicked)
        self.update()

    def change_method(self, method):
        self.method = method
        if self.method == "raw":
            self.console.pars_button.setEnabled(False)
        else:
            self.console.pars_button.setEnabled(True)
        self.reset_args()
        self.get_clustering(self.args[self.method])

    def set_method(self, ev):
        if not ev:
            self.video_mode = "same"
            self.open_videos = {}
        else:
            self.video_mode = "new"

    def close_all_videos(self):
        for n in list(self.open_videos.keys()):
            self.open_videos[n].close()

    def set_rect(self, event):
        cur_x, cur_y, start_x, start_y = event
        self.rect_position = [start_x, start_y, cur_x - start_x, cur_y - start_y]
        self.plot_clusters()

    def on_release(self):
        s_x, s_y, w, h = self.rect_position
        e_x = s_x + w
        e_y = s_y + h
        x = (self.components[:, 0] > min(s_x, e_x)) & (
            self.components[:, 0] < max(s_x, e_x)
        )
        y = (self.components[:, 1] > min(s_y, e_y)) & (
            self.components[:, 1] < max(s_y, e_y)
        )
        if self.graph.add:
            self.chosen += list(np.where(x & y)[0])
        else:
            self.chosen = list(np.where(x & y)[0])
        self.rect_position = [0, 0, 0, 0]
        self.plot_clusters()
        self.console.save_button.setDisabled(len(self.chosen) == 0)
        self.console.open_button.setDisabled(len(self.chosen) == 0)


    def set_select_mode(self):
        self.select_mode = True
        self.graph.select_mode = True

    def set_move_mode(self):
        self.select_mode = False
        self.graph.select_mode = False

    def save(self):
        self.output_file = QFileDialog.getSaveFileName(self, "Save file")[0]
        with open(self.output_file, "wb") as f:
            pickle.dump(self.chosen, f)

    def open_intervals(self, intervals=None, suggestions_folder=None, sort_intervals=False):
        self.close()
        if intervals is None or not isinstance(intervals, Iterable):
            intervals = np.array(self.frames)[self.chosen] # (video, start, end, clip)
        al_dict = None
        suggestion_files = None
        if al_dict is None:
            al_dict = defaultdict(lambda: [])
            for video, start, end, clip in intervals:
                al_dict[video].append([int(start), int(end), str(clip)])
        if suggestions_folder is not None:
            files = os.listdir(suggestions_folder)
            for file in files:
                if file.endswith("al_points.pickle"):
                    break
            if os.path.exists(os.path.join(suggestions_folder, file)):
                with open(os.path.join(suggestions_folder, file), "rb") as f:
                    al_dict = pickle.load(f)
            keys = list(al_dict.keys())
            for key in keys:
                if len(al_dict[key]) == 0:
                    al_dict.pop(key)
            suggestion_files = []
            files = os.listdir(suggestions_folder)
            for fn in self.filenames:
                video_id = fn.split('.')[0]
                if video_id not in al_dict:
                    continue
                for file in files:
                    if file.startswith(video_id):
                        suggestion_files.append(os.path.join(suggestions_folder, file))
        videos = [os.path.join(fp, fn) for fp, fn in zip(self.filepaths, self.filenames) if fn.split(".")[0] in al_dict]
        annotation_files = [ann for ann, fn in zip(self.annotation_files, self.filenames) if fn.split(".")[0] in al_dict]
        if sort_intervals:
            if suggestion_files is not None:
                iterator = zip(videos, suggestion_files)
            else:
                iterator = zip(videos, annotation_files)
            for file, ann in iterator:
                video = os.path.basename(file).split('.')[0]
                if video in al_dict:
                    labels_dict = defaultdict(lambda: [])
                    annotation = Annotation(ann)
                    for start, end, clip in al_dict[video]:
                        labels_dict[annotation.main_labels(start, end, clip)].append([start, end, clip])
                    al_dict[video] = []
                    for label in sorted(labels_dict.keys()):
                        al_dict[video] += labels_dict[label]
        window = annotator.MainWindow(
            videos=videos,
            multiview=False,
            active_learning=True,
            al_points_dictionary=al_dict,
            clustering_parameters=self.parameters,
            config_file=self.settings_file,
            skeleton_files=self.skeleton_files,
            annotation_files=self.annotation_files,
            suggestion_files=suggestion_files,
            hard_negatives="all",
        )
        window.show()

    def open_dlc2action_project(self):
        from dlc2action.project import Project
        return Project(name=self.dlc2action_name, projects_path=self.dlc2action_path)

    def get_behaviors(self):
        behaviors = set()
        for file in os.listdir(self.annotation_folder):
            if not file.endswith(self.settings["suffix"]):
                continue
            with open(os.path.join(self.annotation_folder, file), "rb") as f:
                data = pickle.load(f)
            file_behaviors = [x for i, x in enumerate(data[1]) if any([len(y[i]) > 0 for y in data[3]])]
            file_behaviors = [x for x in file_behaviors if not x.startswith("negative") and not x.startswith("unknown")]
            behaviors.update(file_behaviors)
        return sorted(behaviors)

    def get_episode(self, behaviors):
        episode_name = EpisodeSelector(self.open_dlc2action_project()).exec_()
        if episode_name is None:
            (
                episode_name,
                load_episode,
                num_epochs,
                behaviors
            ) = EpisodeParamsSelector(self.open_dlc2action_project(), behaviors).exec_()
            project = self.open_dlc2action_project()
            project.run_episode(
                episode_name,
                load_episode=load_episode,
                force=True,
                parameters_update={
                    "data": {"behaviors": behaviors},
                    "training": {"num_epochs": num_epochs}
                },
            )
        return episode_name

    def autolabel(self):
        self.close()
        del self.data
        behaviors = self.get_behaviors()
        episode_name = self.get_episode(behaviors)
        behaviors = list(self.open_dlc2action_project().get_behavior_dictionary(episode_name).values())
        suggestion_name, suggestion_params = SuggestionParamsSelector(behaviors).exec_()
        self.run_suggestion(suggestion_name, episode_name, suggestion_params)

    def run_suggestion(self, suggestion_name, episode_name, suggestion_params):
        project = self.open_dlc2action_project()
        project.run_suggestion(
            suggestion_name,
            suggestion_episodes=[episode_name],
            parameters_update={
                "general": {"only_load_annotated": False},
                "data": {"filter_annotated": False}
            },
            force=True,
            delete_dataset=True,
            mode="all",
            cut_annotated=True,
            **suggestion_params,
        )
        self.browse_suggestions(suggestion_name)

    def browse_suggestions(self, suggestion_name: str):
        project = self.open_dlc2action_project()
        suggestions_path = os.path.join(project.project_path, "results", "suggestions", suggestion_name)
        self.open_intervals(suggestions_folder=suggestions_path, sort_intervals=True)


def get_file(folder, filename, suffix, filepath):
    res = None
    id = filename.split('.')[0]
    if folder is None:
        folder = filepath
    if suffix is not None:
        feature_path = os.path.join(folder, id + suffix)
        if os.path.exists(feature_path):
            res = feature_path
    return res

@click.option(
    "--video",
    multiple=True,
    help="The video filepath; to include more than one just repeat this option",
)
@click.option(
    "--video_folder",
)
@click.option(
    "--feature_folder",
)
@click.option(
    "--feature_suffix"
)
@click.option(
    "--annotation_folder"
)
@click.option(
    "--skeleton_folder"
)
@click.option(
    "--dlc2action_name"
)
@click.option(
    "--skip_dlc2action",
    is_flag=True,
)
@click.option(
    "--dlc2action_path"
)
@click.option("--open-settings", "-s", is_flag=True, help="Open settings window")
@click.option("--config_file", "-c", default="config.yaml", help="The config file path")
@click.command()
def main(
        video,
        video_folder,
        open_settings,
        config_file,
        feature_folder,
        feature_suffix,
        annotation_folder,
        skeleton_folder,
        dlc2action_name,
        skip_dlc2action,
        dlc2action_path
):

    app = QApplication(sys.argv)
    filenames = []
    filepaths = []
    features = []
    video = list(video)
    if video_folder is not None:
        video += [os.path.join(video_folder, x) for x in os.listdir(video_folder)]
    for f in video:
        filepath, filename = os.path.split(f)
        filepaths.append(filepath)
        filenames.append(filename)
        features.append(get_file(feature_folder, filename, feature_suffix, filepath))
    if feature_folder is None:
        feature_folder = video_folder

    window = MainWindow(
        filenames=filenames,
        filepaths=filepaths,
        open_settings=open_settings,
        config_file=config_file,
        features=features,
        skeleton_folder=skeleton_folder,
        annotation_folder=annotation_folder,
        dlc2action_name=dlc2action_name,
        skip_dlc2action=skip_dlc2action,
        dlc2action_path=dlc2action_path,
        feature_suffix=feature_suffix,
        feature_folder=feature_folder,
    )
    window.show()
    app.exec_()



if __name__ == "__main__":
    main()
