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
import os
from utils import read_video, read_stack, read_skeleton, read_settings
from widgets.viewbox import VideoViewBox
import annotator
from collections import defaultdict


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

        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.method_checkbox)
        self.radio_layout.addWidget(self.close_videos_button)
        self.radio_layout.addSpacing(100)
        self.radio_layout.addWidget(self.select_radio)
        self.radio_layout.addWidget(self.move_radio)
        self.radio_layout.addSpacing(100)
        self.radio_layout.addWidget(self.save_button)
        self.radio_layout.addWidget(self.open_button)

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

    def __init__(self, settings, filenames, filepaths, data_file, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.settings = settings
        self.filepaths = filepaths
        self.filenames = filenames
        self.data_file = data_file
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

    def find_skeleton_files(self):
        self.skeleton_files = []
        if self.settings["DLC_suffix"][0] is not None:
            for i in range(len(self.filepaths)):
                ok = False
                for suffix in self.settings["DLC_suffix"]:
                    path = os.path.join(
                        self.filepaths[i], self.filenames[i].split(".")[0] + suffix
                    )
                    if os.path.exists(path):
                        self.skeleton_files.append(path)
                        ok = True
                if not ok:
                    self.skeleton_files.append(None)
        else:
            self.skeleton_files = [None for _ in range(len(self.filenames))]
        print(f'{self.skeleton_files=}')

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

    def load_skeleton(self):
        self.find_skeleton_files()
        points_df_dict = {}
        for skeleton_file, filename in zip(self.skeleton_files, self.filenames):
            name = filename.split(".")[0]
            points_df = self.load_animals(skeleton_file)
            points_df_dict[name] = points_df
        return points_df_dict

    def load_files(self):
        self.points_df_dict = self.load_skeleton()
        self.stacks = {}
        self.shapes = {}
        for fp, fn in zip(self.filepaths, self.filenames):
            name = fn.split(".")[0]
            stack, shape, _ = read_video(os.path.join(fp, fn))
            self.stacks[name] = stack
            self.shapes[name] = shape

    def video_closed(self, n):
        self.open_videos.pop(n)
        self.plot_clusters()

    def clicked(self, point, ev):
        data_list = list(self.scatter.data)
        points_list = [x[9] for x in data_list]
        n = points_list.index(ev[0])
        if self.video_mode == "same":
            self.close_all_videos()
        vid, start, end, ind = self.frames[n]
        label = self.labels[n]
        if label in self.label_dict:
            label = self.label_dict[label]
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
        if self.data_file is None:
            import random

            X, y = datasets.make_blobs(
                n_samples=100, cluster_std=[1.0, 2.5, 0.5], n_features=5
            )
            self.label_dict = {0: "zero", 1: "one", 2: "two"}
            self.data = X
            self.labels = y

            videos = [f.split(".")[0] for f in self.filenames]
            n = 100
            videos_rand = [videos[random.randint(0, len(videos) - 1)] for _ in range(n)]
            self.frames = [(v, 0, 10, "ind0") for v in videos_rand]
        else:
            with open(self.data_file, "rb") as f:
                self.frames, self.data, self.labels, self.label_dict = pickle.load(f)

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
        if x == -100:
            col = QColor("gray")
            col.setAlphaF(0.2)
        else:
            col = pg.mkColor(self.colors[x % len(self.colors)] * 255)
        if n in self.chosen and x != -100:
            if col == Qt.blue:
                col.setAlphaF(0.1)
            else:
                col.setAlphaF(0.7)
        return QBrush(col)

    def make_pen(self, n):
        if n in self.chosen:
            return QPen(Qt.blue, 0.2)
        else:
            return QPen(Qt.gray, 0.1)

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

    def open_intervals(self):
        intervals = np.array(self.frames)[self.chosen] # (video, start, end, clip)
        al_dict = defaultdict(lambda: [])
        for video, start, end, clip in intervals:
            al_dict[video].append([int(start), int(end), str(clip)])
        self.close()
        print(f'{al_dict=}')
        window = annotator.MainWindow(
            videos=[os.path.join(fp, fn) for fp, fn in zip(self.filepaths, self.filenames) if fn.split(".")[0] in al_dict],
            multiview=False,
            active_learning=True,
            al_points_dictionary=al_dict
        )
        window.show()


@click.option(
    "--video_files",
    "-v",
    multiple=True,
    required=True,
    help="The video filepath; to include more than one just repeat this option",
)
@click.option(
    "--data_file",
    "-d",
    help="The path to a pickled data file, in the format of (frames, data, labels), where "
    "frames is an N-length list of (video_id, start, end, individual) tuples for the "
    "clips that you are clustering; data is a (N, M)-shape numpy array that contains "
    "the features of those clips; labels is a (N, 1)-shape numpy array that contains "
    "the corresponding labels (-100 stands for no label)",
)
@click.command()
def main(video_files, data_file):

    app = QApplication(sys.argv)
    settings = read_settings("config.yaml")
    filenames = []
    filepaths = []
    for f in video_files:
        filepath, filename = os.path.split(f)
        filepaths.append(filepath)
        filenames.append(filename)
    window = MainWindow(settings, filenames, filepaths, data_file)
    window.show()
    app.exec_()



if __name__ == "__main__":
    main()
