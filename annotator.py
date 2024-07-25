#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import os
import pickle
import sys
import yaml

from pathlib import Path
from typing import Optional

import click
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from utils import get_settings, read_settings, read_video, save_settings
from widgets.core.backup import BackupManager
from widgets.dialog import Form, FormInit
from widgets.settings import SettingsWindow, SetNewProject
from widgets.viewer import Viewer
from widgets.viewer_act_list import ViewerActList


def select_folder():
    options = QFileDialog.Options()
    options |= QFileDialog.ShowDirsOnly
    folder_path = QFileDialog.getExistingDirectory(
        None, "Select Directory", "", options=options
    )

    if folder_path:
        return folder_path


def create_load():
    """Create a new project or load an existing project, returns the project folder path"""

    start_dialog = QMessageBox()
    start_dialog.setText("Welcome to DLC2action! ")

    create_project = start_dialog.addButton("Create Project", QMessageBox.ActionRole)
    open_project = start_dialog.addButton("Open Project", QMessageBox.ActionRole)

    start_dialog.exec_()

    if start_dialog.clickedButton() == create_project:
        newProject = SetNewProject()
        newProject.exec_()
        
        folder_path = newProject.folder_path
    elif start_dialog.clickedButton() == open_project:
        folder_path = select_folder()

    return folder_path


class MainWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(
        self,
        folder_path: str,
        backup_dir: Optional[str] = None,
        backup_interval: int = 30,
    ):

        super(MainWindow, self).__init__()

        self.toolbar = None
        self.menubar = None
        self.viewer: Optional[Viewer] = None

        # Backup parameters
        self.backup_manager: Optional[BackupManager] = None
        self.backup_dir = backup_dir
        self.backup_interval = backup_interval

        # Video parameters
        self.cur_video = 0
        self.clustering_parameters = None
        self.root_directory = folder_path
        self.multiview = False
        self.dev = False
        # Modes
        self.sequential = False
        self.current_folder = None
        self.al_mode = True
        self.al_points_file = None
        # self.al_points_dict = al_points_dictionary
        self.al_points_dict = {}

        self.read_settings()
        self.load_project(folder_path=folder_path)

        self.run_video(self.multiview)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.launch_project()

    def read_settings(self):
        self.settings_file = os.path.join(
            self.root_directory, "Project_Config", "config.yaml"
        )
        self.settings = read_settings(self.settings_file)

    def closeEvent(self, a0) -> None:
        print("closeEvent(self, a0) -> None:")

        if self.backup_manager:
            self.backup_manager.stop()
        # self.backup_manager.stop()
        self.closed.emit()
        super().closeEvent(a0)

    def next_video(self):
        if (
            self.clustering_parameters is not None
            and self.cur_video == len(self.videos) - 1
        ):
            self.close()
        else:
            self.cur_video = (self.cur_video + 1) % len(self.videos)
            self.settings = read_settings(self.settings_file)
            self.run_viewer_single()

    def prev_video(self):
        self.cur_video = self.cur_video - 1
        if self.cur_video < 0:
            self.cur_video = len(self.videos) + self.cur_video
        self.run_viewer_single()

    # def load_project(self, videos, annotation_files, suggestion_files, hard_negatives):
    def load_project(self, folder_path=None):

        if folder_path is None:
            self.folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")
        else:
            self.folder_path = folder_path

        # Get the list of folders in the selected directory
        folders = ["Annotations", "Project_Config", "Tracking data"]
        num_videos = len(os.listdir(os.path.join(self.folder_path, "Tracking data")))

        videos = self.settings["video_files"]
        annotation_files = self.settings["skeleton_files"]
        suggestion_files = self.settings["suggestion_files"]
        hard_negatives = self.settings["hard_negative_classes"]

        for folder_name in folders:
            if folder_name == "Annotations":

                if annotation_files is None:
                    annotation_files = [None for _ in num_videos]
                self.annotation_files = annotation_files

            elif folder_name == "Project_Config":

                self.settings_file = os.path.join(
                    self.folder_path, "Project_Config", "config.yaml"
                )
                self.settings = get_settings(self.settings_file, show_settings=False)
                skeleton = self.settings["skeleton_files"]
                with open("colors.txt") as f:
                    self.animal_colors = [
                        list(map(lambda x: float(x) / 255, line.split()))
                        for line in f.readlines()
                    ]

            elif folder_name == "Tracking data":

                # Get the list of files in Tracking data folder
                folder_path = os.path.join(self.folder_path, "Tracking data")
                files = os.listdir(folder_path)

                # Filter video files based on extensions
                self.videos = [
                    os.path.join(folder_path, file)
                    for file in files
                    if file.lower().endswith((".mov", ".avi", ".mp4", ".mkv"))
                ]

                # Check the number of video files
                num_videos = len(self.videos)

                # Perform actions based on the number of videos
                if num_videos > 1:
                    msg = QMessageBox()
                    msg.setText(
                        "You have chosen more than one video file. Would you like to open them in multiple view mode?"
                    )
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    reply = msg.exec_()
                    if reply == QMessageBox.Yes:
                        self.multiview = True
                    else:

                        self.multiview = False

                if len(videos) == 0 and self.settings["video_files"] is not None:
                    videos = self.settings["video_files"]

                if len(videos) == 0 and self.settings["video_upload_window"]:
                    # This is extra because videos already loaded
                    # self.load_video()
                    pass
                else:
                    if videos == ():
                        self.videos = [None for i in self.settings["skeleton_files"]]
                    else:
                        self.videos = videos
                        if type(self.videos) is not list:
                            self.videos = list(self.videos)

        if annotation_files is None:
            annotation_files = [None for _ in self.videos]
        self.annotation_files = annotation_files

        if suggestion_files is None:
            suggestion_files = [None for _ in self.videos]
        self.suggestion_files = suggestion_files

        os.chdir(self.folder_path)
        self.run_video(self.multiview)

        if hard_negatives is not None:
            self.settings["hard_negative_classes"] = hard_negatives

        return skeleton

    def launch_project(self):
        self._createActions()
        self._createToolBar()
        self._createMenuBar()

    def run_video(self, multiview=False):

        videos = self.videos

        stacks, shapes, lens, filepaths, filenames = [], [], [], [], []
        self.settings = read_settings(self.settings_file)

        print(" self.settings  ", self.settings["skeleton_files"])

        if multiview:
            for i, video in enumerate(videos):
                stack, shape, length = read_video(video, self.settings["backend"])
                stacks.append(stack)
                shapes.append(shape)
                lens.append(length)

                if video is not None:

                    filepath, filename = os.path.split(video)
                    filepaths.append(filepath)
                    filenames.append(filename)

                else:
                    print('self.settings["skeleton_files"][i] ', self.settings)
                    filepath, filename = os.path.split(
                        self.settings["skeleton_files"][i]
                    )
                    filepaths.append(filepath)
                    filenames.append(filename)

            self.run_viewer(
                stacks,
                shapes,
                lens,
                filenames,
                filepaths,
                self.annotation_files[self.cur_video],
                self.suggestion_files,
                current=0,
            )

        else:
            if len(self.videos) > 1:
                self.sequential = True
            self.run_viewer_single(current=0)

    def open_video(self):
        # self.viewer.save()
        self.load_video(self.current_folder)
        # TODO: Bug here when you open a view it doesn't prompt you to open in multiple view
        self.run_video()

    def read_video_stack(self, n):
        stack, shape, length = read_video(self.videos[n], self.settings["backend"])
        stacks = [stack]
        shapes = [shape]
        lens = [length]
        if self.videos[n] is None:
            filepath, filename = os.path.split(self.settings["skeleton_files"][n])
        else:
            filepath, filename = os.path.split(self.videos[n])
        return stacks, shapes, lens, filename, filepath

    def run_viewer_single(self, current=0):
        stacks, shapes, lens, filename, filepath = self.read_video_stack(self.cur_video)
        self.run_viewer(
            stacks,
            shapes,
            lens,
            [filename],
            [filepath],
            self.annotation_files[self.cur_video],
            self.suggestion_files[self.cur_video],
            current,
        )

    def run_viewer(
        self,
        stacks,
        shapes,
        lens,
        filenames,
        filepaths,
        annotation,
        suggestion,
        current,
    ):

        al_points = self.get_al_points(filenames[0])
        if al_points is None:
            self.al_mode = False
        if annotation is None:
            annotation = self.annotation_files[0]
        if annotation is None:
            suggestion = self.suggestion_files[0]

        if self.backup_manager is not None:
            self.backup_manager.stop()

        self.viewer = Viewer(
            stacks,
            shapes,
            lens,
            None,
            annotation,
            suggestion,
            self.settings,
            self.sequential,
            filenames,
            filepaths,
            current,
            al_mode=self.al_mode,
            al_points=al_points,
        )
        self.setCentralWidget(self.viewer)
        self.viewer.status.connect(self.statusBar().showMessage)
        self.viewer.next_video.connect(self.next_video)
        self.viewer.prev_video.connect(self.prev_video)
        self.viewer.mode_changed.connect(self.on_mode)

        if self.backup_dir is None:
            default_video_path = Path(self.videos[0])
            backup_path = default_video_path.with_name(
                default_video_path.stem + "_backups"
            )
        else:
            backup_path = Path(self.backup_dir)
        self.backup_manager = BackupManager(
            backup_path=backup_path,
            viewer=self.viewer,
            interval=self.backup_interval,
        )
        self.backup_manager.start()

    def get_al_points(self, filename):
        if self.dev:
            al_points = [[(107, 283, "ind6"), (222, 388, "ind14"), (226, 258, "ind16")]]
            return al_points[self.cur_video]
        else:
            sep = self.settings["prefix_separator"]
            name = filename.split(".")[0]
            if sep is not None:
                name = sep.join(name.split(sep)[1:])
            if self.al_points_file is not None:
                with open(self.al_points_file, "rb") as f:
                    al_points = pickle.load(f)
            elif self.al_points_dict is not None:
                al_points = self.al_points_dict
            else:
                return None
            return al_points.get(name)

    def load_data_skl(self, skeleton):

        update = False
        settings_update = {}
        self.settings["skeleton_files"] = skeleton
        videos = self.videos

        if skeleton is None or len(skeleton[0]) == 0:
            return None

        elif len(self.videos) > 0:
            # Match skeleton files with corresponding videos

            for skeleton_file in skeleton:
                user_input = FormInit(videos, skeleton_file)
                user_input.exec_()

            updated_skeleton = []
            suffixes = [file.split("/")[-1].split(".")[0] for file in self.videos]
            suffix_to_filename = {}
            for suffix in suffixes:
                for filename in skeleton:
                    if suffix in filename:
                        suffix_to_filename[suffix] = filename
                        break
                else:
                    suffix_to_filename[suffix] = None
            updated_skeleton = [suffix_to_filename[suffix] for suffix in suffixes]
            skeleton = updated_skeleton
            self.settings["skeleton_files"] = updated_skeleton
            settings_update["skeleton_files"] = updated_skeleton
            update = True

        if update:
            self.viewer.save(verbose=False, ask=False)
            print("settings_update ", settings_update)
            self.run_video(
                current=self.viewer.current(),
                settings_update=settings_update,
                multiview=self.multiview,
            )
            self._createActions()
            self._createMenuBar()
            self._createToolBar()

    def load_data(self, type):
        update = False
        settings_update = {}
        if type == "boxes":
            boxes = [QFileDialog.getOpenFileName(self, "Open file")[0]]
            if len(boxes[0]) == 0:
                boxes = None
            if boxes is not None:
                settings_update["detection_files"] = boxes
                update = True
        if type == "DLC":

            skeleton = [
                QFileDialog.getOpenFileName(
                    self, "Open file", filter="DLC files (*.h5 *.pickle)"
                )[0]
            ]
            if len(skeleton[0]) == 0:
                skeleton = None
            elif len(self.videos) > 0:
                video = Form(self.videos).exec_()
                if self.settings["skeleton_files"] == [None]:
                    files = [None for _ in self.videos]
                else:
                    files = self.settings["skeleton_files"]
                skeleton = [
                    files[i] if x != video else skeleton[0]
                    for i, x in enumerate(self.videos)
                ]
            if skeleton is not None:
                settings_update["skeleton_files"] = skeleton
                update = True
        # if type == "load_labels_file":
        #     load_labels_file = QFileDialog.getOpenFileName(
        #         self, "Open file", filter="Annotation files (*.h5 *.pickle)"
        #     )[0]
        #     if load_labels_file != "":
        #         self.load_labels_file = load_labels_file
        #         update = True
        if update:
            self.viewer.save(verbose=False, ask=False)
            self.run_video(
                current=self.viewer.current(),
                settings_update=settings_update,
                multiview=self.multiview,
            )
            self._createActions()
            self._createMenuBar()
            self._createToolBar()

    def setRootDirectory(self):
        print("os.chdir(self.root_directory) ", self.root_directory)
        os.chdir(self.root_directory)

    def _createActions(self):
        self.setRootDirectory()
        print("curr dir ", os.getcwd())
        # File actions
        self.play_action = QAction(self)
        self.play_action.setText("Play / Stop")
        self.play_action.setIcon(QIcon("icons/pause-button.png"))
        self.play_action.triggered.connect(lambda: self.viewer.on_play())
        self.move_action = QAction(self, checkable=True)
        self.move_action.setChecked(True)
        self.move_action.setText("Move")
        self.move_action.setShortcut("Ctrl+M")
        self.move_action.setIcon(QIcon("icons/hand.png"))
        self.move_action.triggered.connect(self.viewer.set_move_mode)
        self.remove_action = QAction(self, checkable=True)
        self.remove_action.setText("Remove")
        self.remove_action.setIcon(QIcon("icons/trash.png"))
        self.remove_action.triggered.connect(self.viewer.set_remove_mode)
        self.remove_action.setShortcut("Ctrl+R")
        self.new_action = QAction(self, checkable=True)
        self.new_action.setText("New")
        self.new_action.triggered.connect(self.viewer.set_new_mode)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setIcon(QIcon("icons/plus.png"))
        self.cut_action = QAction(self, checkable=True)
        self.cut_action.setText("Cut")
        self.cut_action.setIcon(QIcon("icons/scissors.png"))
        self.cut_action.triggered.connect(self.viewer.set_cut_mode)
        self.cut_action.setShortcut("Ctrl+C")
        self.ass_action = QAction(self, checkable=True)
        self.ass_action.setText("Assign")
        self.ass_action.setIcon(QIcon("icons/pantone.png"))
        self.ass_action.triggered.connect(self.viewer.set_ass_mode)
        self.ass_action.setShortcut("Ctrl+A")
        self.amb_action = QAction(self, checkable=True)
        self.amb_action.setText("Ambiguous")
        self.amb_action.setIcon(QIcon("icons/transparency.png"))
        self.amb_action.triggered.connect(self.viewer.set_amb_mode)
        self.amb_action.setShortcut("Ctrl+B")

        group = QActionGroup(self)
        group.addAction(self.move_action)
        group.addAction(self.remove_action)
        group.addAction(self.amb_action)
        group.addAction(self.ass_action)
        group.addAction(self.new_action)
        group.addAction(self.cut_action)

    def _createToolBar(self):
        # File toolbar
        if self.toolbar is not None:
            self.toolbar.close()
        self.toolbar = self.addToolBar("Modes")
        self.toolbar.addAction(self.play_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.move_action)
        self.toolbar.addAction(self.remove_action)
        self.toolbar.addAction(self.cut_action)
        self.toolbar.addAction(self.new_action)
        self.toolbar.addAction(self.amb_action)
        self.toolbar.addAction(self.ass_action)

    def _createMenuBar(self):
        if self.menubar is not None:
            self.menubar.clear()
        saveAction = QAction("&Save", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.setStatusTip("Save annotation")
        saveAction.triggered.connect(lambda: self.viewer.save())
        saveasAction = QAction("&Save as...", self)
        saveasAction.setStatusTip("Save annotation to a new file")
        saveasAction.triggered.connect(lambda: self.viewer.save(new_file=True))
        openVideoAction = QAction("&Open...", self)
        openVideoAction.setStatusTip("Open another video")
        openVideoAction.triggered.connect(self.open_video)
        loadDLCAction = QAction("&DLC...", self)
        loadDLCAction.setStatusTip("Load DLC output")
        loadDLCAction.triggered.connect(lambda: self.load_data(type="DLC"))
        loadLabelAction = QAction("&Annotation...", self)
        loadLabelAction.setStatusTip("Load an annotation file")
        loadLabelAction.triggered.connect(lambda: self.load_data(type="labels"))
        loadFromListAction = QAction("&Load from list...", self)
        loadFromListAction.setStatusTip(
            "Load labels from the settings file list and switch to/from nested annotation"
        )
        loadFromListAction.triggered.connect(self.viewer.load_cats)

        changeLabelsAction = QAction("&Change labels...", self)
        changeLabelsAction.setShortcut("Ctrl+L")  # Set the shortcut for changing labels
        changeLabelsAction.setStatusTip("Modify the label names and the shortcuts")
        changeLabelsAction.triggered.connect(self.viewer.get_cats)
        activeLearningAction = QAction("&Start/Stop active learning", self)
        activeLearningAction.setStatusTip(
            "Activate or deactivate the more efficient active learning mode"
        )
        activeLearningAction.triggered.connect(self.change_al_mode)
        unlabeledAction = QAction("&Start unlabeled search", self)
        unlabeledAction.setStatusTip(
            "Navigate through the unlabeled intervals in the video"
        )
        unlabeledAction.triggered.connect(self.set_unlabeled_al)
        labelAction = QAction("&Start label search", self)
        labelAction.setStatusTip(
            "Find the intervals annotated with a specific label in the video"
        )
        labelAction.triggered.connect(self.set_label_al)
        trackletAction = QAction("&Start tracklet navigation", self)
        trackletAction.setStatusTip("Go through tracklets one by one")
        trackletAction.triggered.connect(self.set_tracklet_al)

        rainbowAction = QAction("&Body part colors", self)
        rainbowAction.setStatusTip("Switch between individual and body part coloring")
        rainbowAction.triggered.connect(self.viewer.switch_rainbow)
        skeletonAction = QAction("&Skeleton", self)
        skeletonAction.setStatusTip("Display skeleton connections")
        skeletonAction.triggered.connect(self.viewer.switch_skeleton)
        if self.settings["skeleton"] is None or len(self.settings["skeleton"]) == 0:
            skeletonAction.setDisabled(True)
        reprAction = QAction("&Reprojections", self)
        reprAction.setStatusTip("Display reprojection points")
        reprAction.triggered.connect(self.viewer.switch_repr)
        if (
            self.settings["3d_suffix"] is None
            or self.settings["calibration_path"] is None
        ):
            reprAction.setDisabled(True)
        assAction = QAction("&Assess suggestions...", self)
        assAction.setStatusTip("Sample suggestions and assess them as true or false")
        assAction.triggered.connect(self.set_assessment_al)
        settingsAction = QAction("&Open settings", self)
        settingsAction.setStatusTip("Open the settings window")
        settingsAction.triggered.connect(self.set_settings)

        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)
        file = self.menubar.addMenu("File")

        # TODO: Does this save the project?
        file.addAction(saveAction)
        file.addAction(saveasAction)

        file.addAction(openVideoAction)

        # file.addAction(correctAction)
        # file.addAction(exampleAction)
        loadMenu = file.addMenu("Load")
        loadMenu.addAction(loadDLCAction)
        loadMenu.addAction(loadLabelAction)
        labels = self.menubar.addMenu("Labels")
        labels.addAction(changeLabelsAction)
        labels.addAction(loadFromListAction)
        manual = self.menubar.addMenu("Manual")
        manual.addAction(self.move_action)
        manual.addAction(self.remove_action)
        manual.addAction(self.new_action)
        manual.addAction(self.cut_action)
        manual.addAction(self.ass_action)
        manual.addAction(self.amb_action)
        al = self.menubar.addMenu("Active learning")
        al.addAction(activeLearningAction)
        al.addAction(unlabeledAction)
        al.addAction(labelAction)
        al.addAction(trackletAction)
        al.addAction(assAction)
        display = self.menubar.addMenu("Display")
        display.addAction(rainbowAction)
        display.addAction(skeletonAction)
        display.addAction(reprAction)
        settings = self.menubar.addMenu("Settings")
        settings.addAction(settingsAction)

    def on_mode(self, mode):
        modes = {
            "As": self.ass_action,
            "A": self.amb_action,
            "R": self.remove_action,
            "M": self.move_action,
            "C": self.cut_action,
            "N": self.new_action,
        }
        if not modes[mode].isChecked():
            modes[mode].setChecked(True)

    def change_al_mode(self, event):
        filename = self.viewer.filenames[0]
        if self.settings["prefix_separator"] is not None:
            sep = self.settings["prefix_separator"]
            split = filename.split(sep)
            if len(split) > 1:
                filename = sep.join(split[1:])
        if not self.al_mode and self.get_al_points(filename) is None:
            self.viewer.show_warning(
                "The active learning file does not have information for this video!"
            )
            return
        else:
            self.viewer.al_points = self.get_al_points(filename)
        self.al_mode = not self.al_mode
        self.viewer.change_al_mode(self.al_mode)

    def set_unlabeled_al(self, event):
        self.viewer.set_unlabeled_al()
        self.al_mode = True
        self.viewer.change_al_mode(self.al_mode)

    def set_tracklet_al(self, event):
        self.viewer.set_tracklet_al()
        self.al_mode = True
        self.viewer.change_al_mode(self.al_mode)

    def set_label_al(self, event):
        ok = self.viewer.set_label_al()
        if ok:
            self.al_mode = True
            self.viewer.change_al_mode(self.al_mode)

    def set_assessment_al(self, event):
        self.viewer.set_assessment_al()

    def set_settings(self, event):
        SettingsWindow(self.settings_file).exec_()
        self.viewer.save(verbose=False, ask=False)
        self.run_video(current=self.viewer.current(), multiview=self.multiview)
        self._createActions()
        self._createToolBar()
        self._createMenuBar()


@click.command()
@click.option(
    "--video",
    multiple=True,
    help="The video file to annotate (for more views repeat several times)",
)
@click.option(
    "--multiview",
    "-m",
    is_flag=True,
    help="Display multiple videos in parallel (when False multiple videos will be displayed sequentially)",
)
@click.option(
    "--dev",
    "-d",
    is_flag=True,
    help="Development mode (some artificial debugging settings)",
)
@click.option("--active_learning", "-a", is_flag=True, help="Active learning mode")
@click.option(
    "--backup-dir", "-b", default=None, help="The directory where backups are saved"
)
@click.option(
    "--backup-interval",
    default=30,
    type=int,
    help="The interval between backups, in minutes",
)
def main(video, multiview, dev, active_learning, backup_dir, backup_interval):
    app = QApplication(sys.argv)
    folder_path = create_load()

    window = MainWindow(
        folder_path=folder_path, backup_dir=backup_dir, backup_interval=backup_interval
    )

    window.show()
    app.exec_()


if __name__ == "__main__":

    main()
