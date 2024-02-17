#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import os
import yaml
import numpy as np
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtwidgets import Toggle
from ruamel.yaml import YAML


class MultipleInputWidget(QWidget):
    def __init__(self, values):
        super(MultipleInputWidget, self).__init__()
        self.setMinimumWidth(300)
        self.layout = QVBoxLayout()
        listBox = QVBoxLayout(self)
        self.setLayout(listBox)

        scroll = QScrollArea(self)
        listBox.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)

        scrollContent.setLayout(self.layout)
        scroll.setWidget(scrollContent)
        self.lines_layout = QVBoxLayout()
        self.lines = []
        if isinstance(values, str):
            values = [values]
        for value in values:
            self.lines.append(QLineEdit(value))
            self.lines_layout.addWidget(self.lines[-1])
            self.lines[-1].setMinimumHeight(20)
        self.button = QPushButton("Add")
        self.button.clicked.connect(self.add_line)
        self.layout.addLayout(self.lines_layout)
        self.layout.addWidget(self.button)

    def add_line(self, value=None):
        self.lines.append(QLineEdit())
        self.lines_layout.addWidget(self.lines[-1])
        self.lines[-1].setMinimumHeight(20)

    def values(self):
        return [line.text() for line in self.lines if line.text() != ""]

class MultipleDoubleInputWidget(QWidget):
    def __init__(self, values):
        super(MultipleDoubleInputWidget, self).__init__()
        self.setMinimumWidth(300)
        self.layout = QVBoxLayout()
        listBox = QVBoxLayout(self)
        self.setLayout(listBox)

        scroll = QScrollArea(self)
        listBox.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)

        scrollContent.setLayout(self.layout)
        scroll.setWidget(scrollContent)
        self.lines_layout = QVBoxLayout()
        self.lines = []
        for value1, value2 in values:
            self.lines.append([QLineEdit(value1), QLineEdit(value2)])
            line = QHBoxLayout()
            line.addWidget(self.lines[-1][0])
            line.addWidget(QLabel(", "))
            line.addWidget(self.lines[-1][1])
            self.lines_layout.addLayout(line)
        self.button = QPushButton("Add")
        self.button.clicked.connect(self.add_line)
        self.layout.addLayout(self.lines_layout)
        self.layout.addWidget(self.button)

    def add_line(self, value=None):
        self.lines.append([QLineEdit(), QLineEdit()])
        line = QHBoxLayout()
        line.addWidget(self.lines[-1][0])
        line.addWidget(QLabel(", "))
        line.addWidget(self.lines[-1][1])
        self.lines_layout.addLayout(line)

    def values(self):
        return [
            [line[0].text(), line[1].text()]
            for line in self.lines
            if line[0].text() != "" and line[1].text() != ""
        ]

class CategoryInputWidget(QWidget):
    def __init__(self, values):
        super(CategoryInputWidget, self).__init__()
        self.setMinimumWidth(300)
        self.layout = QVBoxLayout()
        listBox = QVBoxLayout(self)
        self.setLayout(listBox)

        scroll = QScrollArea(self)
        listBox.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)

        scrollContent.setLayout(self.layout)
        scroll.setWidget(scrollContent)
        self.lines_layout = QVBoxLayout()
        self.lines = []
        for key, value in values.items():
            line_text = ",  ".join(value)
            self.lines.append([QLineEdit(key), QLineEdit(line_text)])
            self.lines[-1][0].setMinimumWidth(90)
            self.lines[-1][1].setMinimumWidth(150)
            line = QHBoxLayout()
            line.addWidget(self.lines[-1][0])
            line.addWidget(QLabel(": "))
            line.addWidget(self.lines[-1][1])
            self.lines_layout.addLayout(line)
        self.button = QPushButton("Add")
        self.button.clicked.connect(self.add_line)
        self.layout.addLayout(self.lines_layout)
        self.layout.addWidget(self.button)

    def add_line(self, value=None):
        self.lines.append([QLineEdit(), QLineEdit()])
        self.lines[-1][0].setMinimumWidth(90)
        self.lines[-1][1].setMinimumWidth(150)
        line = QHBoxLayout()
        line.addWidget(self.lines[-1][0])
        line.addWidget(QLabel(": "))
        line.addWidget(self.lines[-1][1])
        self.lines_layout.addLayout(line)

    def values(self):
        return {
            line[0].text(): list(map(lambda x: x.strip(), line[1].text().split(",")))
            for line in self.lines
            if line[0].text() != "" and line[1].text() != ""
        }

class SettingsWindow(QDialog):
    def __init__(self, config_path):
        super(SettingsWindow, self).__init__()
        self.config_path = config_path
        self.settings = self._open_yaml(config_path)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.labels = {}
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.tabs = QTabWidget()
        self.tabs.tabBarClicked.connect(self.collect)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.functions = [
            self.collect_general,
            self.collect_display,
            self.collect_al,
            self.collect_fp,
        ]
        self.create_general_tab()
        self.set_general_tab()
        self.create_display_tab()
        self.set_display_tab()
        self.create_al_tab()
        self.set_al_tab()
        self.create_fp_tab()
        self.set_fp_tab()

    def update_data(self):
        self.set_general_tab()
        self.set_display_tab()
        self.set_al_tab()
        self.set_fp_tab()

    def collect(self, event=None):
        for func in self.functions:
            func()
        self.update_data()

    def set_le(self, field, set_int=True):
        le = QLineEdit()
        if set_int:
            le.setValidator(QIntValidator())
        le.setText(str(self.settings[field]))
        return le

    def set_combo(self, field, options):
        combo = QComboBox()
        for o in options:
            combo.addItem(o)
        combo.setCurrentIndex(options.index(self.settings[field]))
        return combo

    def create_al_tab(self):
        self.al_tab = QWidget()
        self.tabs.addTab(self.al_tab, "Active learning")
        self.al_layout = QFormLayout()
        self.al_tab.setLayout(self.al_layout)

    def set_al_tab(self):
        self.clearLayout(self.al_layout)
        self.set_al_tab_data()
        self.al_layout.addRow("Max loaded frames (AL): ", self.max_loaded_al_le)
        self.al_layout.addRow("Load chunk (AL) (frames): ", self.load_chunk_al_le)
        self.al_layout.addRow("Load buffer (AL) (frames): ", self.load_buffer_al_le)
        self.al_layout.addRow("Number of AL windows: ", self.al_window_num_le)
        self.al_layout.addRow("AL points file: ", self.al_points_file_w)
        self.al_layout.addRow("AL play buffer: ", self.al_buffer_le)
        self.al_layout.addRow("Hard negative classes: ", self.hn_ms)
        self.al_layout.addRow("Number of assessment samples: ", self.assessment_n_le)

    def set_al_tab_data(self):
        self.max_loaded_al_le = self.set_le("max_loaded_frames_al")
        self.load_chunk_al_le = self.set_le("load_chunk_al")
        self.load_buffer_al_le = self.set_le("load_buffer_al")
        self.al_window_num_le = self.set_le("al_window_num")
        self.al_points_file_w = self.set_file("al_points_file", "PICKLE (*.pickle)")
        self.al_buffer_le = self.set_le("al_buffer")
        self.hn_ms = QListWidget()
        actions = []
        for value_list in self.settings["actions"].values():
            actions += value_list
        actions = sorted(set(actions))
        self.hn_ms.addItems(actions)
        self.hn_ms.setSelectionMode(QListWidget.MultiSelection)
        index = self.settings["hard_negative_classes"]
        if index is None:
            index = []
        for i in index:
            matching_items = self.hn_ms.findItems(i, Qt.MatchExactly)
            for item in matching_items:
                item.setSelected(True)
        self.assessment_n_le = self.set_le("assessment_n")

    def set_file(self, field, filter=None, dir=False):
        layout = QHBoxLayout()
        file = self.settings[field]
        file = file if file is not None else "None"
        button = QPushButton("Find")
        label = QLabel(os.path.basename(file))
        if dir:
            button.clicked.connect(lambda: self.get_dir(label, field, filter))
        else:
            button.clicked.connect(lambda: self.get_file(label, field, filter))
        layout.addWidget(label)
        layout.addWidget(button)
        return layout

    def set_spinbox(self, field, minimum, maximum, singlestep=None):
        box = QSpinBox()
        box.setMaximum(maximum)
        box.setMinimum(minimum)
        if singlestep is not None:
            box.setSingleStep(singlestep)
        box.setValue(self.settings[field])
        return box

    def set_toggle(self, field):
        toggle = Toggle()
        toggle.setChecked(self.settings[field])
        return toggle

    def set_slider(self, field, minimum, maximum, percent=False):
        value = self.settings[field]
        if percent:
            minimum *= 100
            maximum *= 100
            value *= 100
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        return slider

    def set_multiple_input(self, field, type="single"):
        if self.settings[field] is None:
            if type == "category":
                x = {}
            else:
                x = []
        else:
            x = self.settings[field]
        if type == "double":
            widget = MultipleDoubleInputWidget(x)
        elif type == "category":
            widget = CategoryInputWidget(x)
        else:
            widget = MultipleInputWidget(x)
        return widget

    def get_file(self, label_widget, field, filter=None):
        file = QFileDialog().getOpenFileName(self, filter=filter)[0]
        label_widget.setText(os.path.basename(file))
        self.settings[field] = file

    def get_dir(self, label_widget, field, filter=None):
        file = QFileDialog().getExistingDirectory(self)
        label_widget.setText(os.path.basename(file))
        self.settings[field] = file

    def create_general_tab(self):
        self.general_tab = QWidget()
        self.tabs.addTab(self.general_tab, "General")
        self.general_layout = QFormLayout()
        self.general_tab.setLayout(self.general_layout)

    def set_general_tab(self):
        self.clearLayout(self.general_layout)
        self.set_general_tab_data()
      
        self.general_layout.addRow("Annotator name: ", self.annotator)
        self.general_layout.addRow("Behaviors: ", self.behaviors)
        self.general_layout.addRow("Data type: ", self.data_type_combo)
        self.general_layout.addRow("Number of individuals: ", self.n_ind_le)
        self.general_layout.addRow("Max number of frames in RAM: ", self.max_loaded_le)
        self.general_layout.addRow("Load chunk (frames): ", self.chunk_le)
        self.general_layout.addRow("Load buffer (frames): ", self.buffer_le)
        self.general_layout.addRow(
            "Minimum tracklet length (frames): ", self.min_frames_le
        )

    def set_general_tab_data(self):

        self.annotator = self.set_le("annotator", set_int=False)
        self.project = self.set_le("project", set_int=False)
        self.behaviors = self.set_multiple_input("actions", type="category")
        self.data_type_combo = self.set_combo("data_type", ["dlc", "calms21"])
        self.n_ind_le = self.set_le("n_ind")
        self.max_loaded_le = self.set_le("max_loaded_frames")
        self.chunk_le = self.set_le("load_chunk")
        self.buffer_le = self.set_le("load_buffer")
        self.min_frames_le = self.set_le("min_length_frames")

    def create_display_tab(self):
        self.display_tab = QWidget()
        self.tabs.addTab(self.display_tab, "Display")
        self.display_layout = QFormLayout()
        self.display_tab.setLayout(self.display_layout)

    def set_display_tab(self):
        self.clearLayout(self.display_layout)
        self.set_display_tab_data()
        self.display_layout.addRow("Skeleton marker size: ", self.skeleton_size_slider)
        self.display_layout.addRow("Console width: ", self.console_width_slider)
        self.display_layout.addRow("Action bar width: ", self.actionbar_width_slider)
        self.display_layout.addRow(
            "Default update frequency: ", self.default_freq_slider
        )
        self.display_layout.addRow("Backend: ", self.backend_combo)
        self.display_layout.addRow("Canvas size: ", self.canvas_size_le)
        self.display_layout.addRow(
            "Detection update frequency: ", self.detection_update_freq_slider
        )
        self.display_layout.addRow("Mask opacity: ", self.mask_opacity_slider)
        self.display_layout.addRow(
            "Loading segmentation policy: ", self.load_segmentation_combo
        )
        self.display_layout.addRow("Likelihood cutoff: ", self.likelihood_cutoff_slider)
        self.display_layout.addRow("3D bodyparts: ", self.bp_3d)
        self.display_layout.addRow("Skeleton edges: ", self.skeleton)

    def set_display_tab_data(self):
        self.skeleton_size_slider = self.set_spinbox("skeleton_size", 0, 7)
        self.console_width_slider = self.set_spinbox("console_width", 100, 500, 25)
        self.actionbar_width_slider = self.set_spinbox("actionbar_width", 50, 400, 25)
        self.default_freq_slider = self.set_spinbox("default_frequency", 10, 100, 10)
        self.backend_combo = self.set_combo(
            "backend", ["pyav_fast", "pyav", "cv2", "decord"]
        )
        self.canvas_size_le = QHBoxLayout()
        self.canvas_size_le_w = QLineEdit(str(self.settings["canvas_size"][0]))
        self.canvas_size_le_w.setValidator(QIntValidator())
        self.canvas_size_le.addWidget(self.canvas_size_le_w)
        self.canvas_size_le.addWidget(QLabel("x"))
        self.canvas_size_le_h = QLineEdit(str(self.settings["canvas_size"][1]))
        self.canvas_size_le_h.setValidator(QIntValidator())
        self.canvas_size_le.addWidget(self.canvas_size_le_h)
        self.detection_update_freq_slider = self.set_spinbox(
            "detection_update_freq", 1, 5
        )
        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider = self.set_slider("mask_opacity", 0, 1, percent=True)
        self.load_segmentation_combo = self.set_combo(
            "load_segmentation", ["ask", "never", "always"]
        )
        self.likelihood_cutoff_slider = QSlider(Qt.Horizontal)
        self.likelihood_cutoff_slider = self.set_slider(
            "likelihood_cutoff", 0, 1, percent=True
        )
        self.bp_3d = self.set_multiple_input("3d_bodyparts")
        self.skeleton = self.set_multiple_input("skeleton", type="double")

    def create_fp_tab(self):
        self.fp_tab = QWidget()
        self.tabs.addTab(self.fp_tab, "File")
        self.fp_layout = QFormLayout()
        self.fp_tab.setLayout(self.fp_layout)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def set_fp_tab(self):
        self.clearLayout(self.fp_layout)
        self.set_fp_tab_data()
        self.fp_layout.addRow("Calibration path: ", self.calibration_path)
        self.fp_layout.addRow("3D pose file suffix: ", self.suffix_3d_le)
        self.fp_layout.addRow("Display 3D pose: ", self.display_3d)
        self.fp_layout.addRow("Display reprojections: ", self.display_repr)
        self.fp_layout.addRow("Annotation file suffix: ", self.suffix_le)
        self.fp_layout.addRow("Prefix separator: ", self.prefix_separator_le)
        self.fp_layout.addRow("Prior annotation file suffix: ", self.prior_suffix_le)
        self.fp_layout.addRow("DLC file suffixes: ", self.dlc_suffix)
        self.fp_layout.addRow("Segmentation suffix: ", self.segmentation_le)

    def set_fp_tab_data(self):
        self.calibration_path = self.set_file("calibration_path", dir=True)
        self.suffix_3d_le = self.set_le("3d_suffix", set_int=False)
        self.display_3d = self.set_toggle("display_3d")
        self.display_repr = self.set_toggle("display_repr")
        self.suffix_le = self.set_le("suffix", set_int=False)
        self.prefix_separator_le = self.set_le("prefix_separator", set_int=False)
        self.prior_suffix_le = self.set_le("prior_suffix", set_int=False)
        self.dlc_suffix = self.set_multiple_input("DLC_suffix")
        self.segmentation_le = self.set_le("segmentation_suffix", set_int=False)

    def collect_general(self):
        self.settings["data_type"] = self.data_type_combo.currentText()
        self.settings["n_ind"] = int(self.n_ind_le.text())
        self.settings["max_loaded_frames"] = int(self.max_loaded_le.text())
        self.settings["load_chunk"] = int(self.chunk_le.text())
        self.settings["load_buffer"] = int(self.buffer_le.text())
        self.settings["actions"] = (
            self.behaviors.values() if len(self.behaviors.values()) > 0 else None
        )
        self.settings["min_length_frames"] = int(self.min_frames_le.text())

    def collect_al(self):
        self.settings["max_loaded_frames_al"] = int(self.max_loaded_al_le.text())
        self.settings["load_chunk_al"] = int(self.load_chunk_al_le.text())
        self.settings["load_buffer_al"] = int(self.load_buffer_al_le.text())
        self.settings["al_window_num"] = int(self.al_window_num_le.text())
        self.settings["al_buffer"] = int(self.al_buffer_le.text())
        self.settings["hard_negative_classes"] = [
            item.text() for item in self.hn_ms.selectedItems()
        ]
        self.settings["assessment_n"] = int(self.assessment_n_le.text())

    def collect_display(self):
        self.settings["backend"] = self.backend_combo.currentText()
        self.settings["skeleton_size"] = self.skeleton_size_slider.value()
        self.settings["console_width"] = self.console_width_slider.value()
        self.settings["actionbar_width"] = self.actionbar_width_slider.value()
        self.settings["default_frequency"] = self.default_freq_slider.value()
        self.settings["canvas_size"] = [
            int(self.canvas_size_le_w.text()),
            int(self.canvas_size_le_h.text()),
        ]
        self.settings[
            "detection_update_freq"
        ] = self.detection_update_freq_slider.value()
        self.settings["mask_opacity"] = self.mask_opacity_slider.value() / 100
        self.settings["load_segmentation"] = self.load_segmentation_combo.currentText()
        self.settings["likelihood_cutoff"] = self.likelihood_cutoff_slider.value() / 100
        self.settings["3d_bodyparts"] = (
            self.bp_3d.values() if len(self.bp_3d.values()) > 0 else None
        )
        self.settings["skeleton"] = (
            self.skeleton.values() if len(self.skeleton.values()) > 0 else None
        )

    def collect_fp(self):
        self.settings["3d_suffix"] = self.suffix_3d_le.text()
        self.settings["suffix"] = self.suffix_le.text()
        self.settings["display_3d"] = self.display_3d.isChecked()
        self.settings["display_repr"] = self.display_repr.isChecked()
        self.settings["prefix_separator"] = self.prefix_separator_le.text()
        self.settings["prior_suffix"] = self.prior_suffix_le.text()
        self.settings["DLC_suffix"] = self.dlc_suffix.values()
        self.settings["annotator"] = self.annotator.text()
        self.settings["project"] = self.project.text()
        self.settings["segmentation_suffix"] = self.segmentation_le.text()

    def accept(self) -> None:
        self.collect()
        for key, value in list(self.settings.items()):
            if value == "None":
                self.settings[key] = None
        with open(self.config_path, "w") as f:
            YAML().dump(self.settings, f)
        if self.settings["suffix"] is None:
            msg = QMessageBox()
            msg.setText(
                "The annotation suffix parameter cannot be None, please set it to a string value!"
            )
            msg.exec_()
            return
        super().accept()

    def _open_yaml(self, path: str):
        """
        Load a parameter dictionary from a .yaml file
        """

        with open(path) as f:
            data = YAML().load(f)
        if data is None:
            data = {}
        return data

# TODO: Delete unused functions 
class SetNewProject(QDialog):
    def __init__(self, config_path):
        super(SetNewProject, self).__init__()
        # TODO: Modify this to be consistent with the creation of a new config file
        self.config_path = config_path
        self.settings = self._open_yaml(config_path)
        # --------------------------------------------
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.labels = {}
        self.folder_name = None  
        self.skeleton = None
        self.buttonBox = QDialogButtonBox(QBtn)
        
        
        self.load_dlcfiles_label = QLabel("Select DLC files (.h5) :")
        self.load_dlcfiles_button = QPushButton("Upload files")
        self.load_dlcfiles_button.clicked.connect(self.load_skeleton)
        self.loaded_dlcfiles_label = QLabel("No video loaded")
        
        self.browse_videos_label = QLabel("Select videos:")
        self.browse_videos_button = QPushButton("Upload videos")
        self.browse_videos_button.clicked.connect(self.load_videos)
        
        self.video_checkbox = QCheckBox("Copy video to folder")
        self.video_checkbox.setChecked(False)
        # Create a QLabel to display the loaded video name
        self.loaded_video_label = QLabel("No video loaded")



        # Connect the Cancel button's rejected signal to close the dialog
        
        # Must correct the reject function
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.accepted.connect(self.accept)
        # --------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.tabBarClicked.connect(self.collect)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        
        self.layout.addWidget(self.load_dlcfiles_label)
        self.layout.addWidget(self.load_dlcfiles_button)
        self.layout.addWidget(self.loaded_dlcfiles_label)
        
        self.layout.addWidget(self.browse_videos_label)
        self.layout.addWidget(self.browse_videos_button)
        self.layout.addWidget(self.loaded_video_label)
        self.layout.addWidget(self.video_checkbox)

      
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        self.functions = [
            self.collect_general,
 
        ]
        self.create_general_tab()
        self.set_general_tab()


            
# TODO: Function structure_project has a larger scope bc it also changes the ui ..
        
    def load_skeleton(self):
        self.skeleton = [QFileDialog.getOpenFileName(self, "Open file")[0]]
        if len(self.skeleton[0]) == 0:
            self.skeleton = None
        else:
            
            file_name = os.path.basename(self.skeleton[0])
            self.loaded_dlcfiles_label.setText(f"Loaded dlc files: {file_name}")
            
        
            
    def load_videos(self, folder = None):


            self.videos = QFileDialog.getOpenFileNames(
                self, "Open file", filter="Video files (*.mov *.avi *mp4 *mkv)"
            )[0]
            
            if type(self.videos) is not list:
                self.videos = [self.videos]
            if len(self.videos) > 1:
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
            

            # Update the text of the loaded_video_label with the name of the first video
            if self.videos:
                video_name = os.path.basename(self.videos[0])
                self.loaded_video_label.setText(f"Loaded video: {video_name}")
            else:
                self.loaded_video_label.setText("No video loaded")
            

        
            
    
    # TODO: How can we change how the files are organised so that we can be in the project folder create 
    # but still access application files 
    def copy_videos_to_tracking_data(self, selected_videos, folder_name, current_directory):
        tracking_data_folder_path = os.path.join(current_directory, folder_name)

        # Create Tracking data folder if it doesn't exist
        if not os.path.exists(tracking_data_folder_path):
            os.makedirs(tracking_data_folder_path)

        # Copy selected videos to Tracking data folder
        for video_path in selected_videos:
            video_filename = os.path.basename(video_path)
            destination_path = os.path.join(tracking_data_folder_path, video_filename)
            shutil.copy2(video_path, destination_path)

        # print("Selected videos copied to 'Tracking data' folder.")
    def copy_skeleton(self, files, current_directory):
        if not files:
            print("The list 'files' is empty.")
        if files: 
            for file_path in files:
                filename = os.path.basename(file_path)
                destination_path = os.path.join(current_directory, filename)
                shutil.copy2(file_path, destination_path)
            
        
    def update_data(self):
        self.set_general_tab()

    def collect(self, event=None):
        for func in self.functions:
            func()
        self.update_data()
        
    def set_le(self, field, set_int=True):
        """
            This function creates a QLineEdit widget, sets it to accept only integers 
            if specified, populates it with a value from the self.settings dictionary 
            based on the provided field, and then returns the configured QLineEdit widget.
        """
        le = QLineEdit()
        if set_int:
            le.setValidator(QIntValidator())
        le.setText(str(self.settings[field]))
        return le

    def set_combo(self, field, options):
        combo = QComboBox()
        for o in options:
            combo.addItem(o)
        combo.setCurrentIndex(options.index(self.settings[field]))
        return combo

    def set_file(self, field, filter=None, dir=False):
        layout = QHBoxLayout()
        file = self.settings[field]
        file = file if file is not None else "None"
        button = QPushButton("Find")
        label = QLabel(os.path.basename(file))
        if dir:
            button.clicked.connect(lambda: self.get_dir(label, field, filter))
        else:
            button.clicked.connect(lambda: self.get_file(label, field, filter))
        layout.addWidget(label)
        layout.addWidget(button)
        return layout

    def set_spinbox(self, field, minimum, maximum, singlestep=None):
        box = QSpinBox()
        box.setMaximum(maximum)
        box.setMinimum(minimum)
        if singlestep is not None:
            box.setSingleStep(singlestep)
        box.setValue(self.settings[field])
        return box

    def set_toggle(self, field):
        toggle = Toggle()
        toggle.setChecked(self.settings[field])
        return toggle

    def set_multiple_input(self, field, type="single"):
        if self.settings[field] is None:
            if type == "category":
                x = {}
            else:
                x = []
        else:
            x = self.settings[field]
        if type == "double":
            widget = MultipleDoubleInputWidget(x)
        elif type == "category":
            widget = CategoryInputWidget(x)
        else:
            widget = MultipleInputWidget(x)
        return widget

    def get_file(self, label_widget, field, filter=None):
        file = QFileDialog().getOpenFileName(self, filter=filter)[0]
        label_widget.setText(os.path.basename(file))
        self.settings[field] = file

    def get_dir(self, label_widget, field, filter=None):
        file = QFileDialog().getExistingDirectory(self)
        label_widget.setText(os.path.basename(file))
        self.settings[field] = file

    def create_general_tab(self):
        self.general_tab = QWidget()
        self.tabs.addTab(self.general_tab, "General")
        self.general_layout = QFormLayout()
        self.general_tab.setLayout(self.general_layout)

    def set_general_tab(self):
        self.clearLayout(self.general_layout)
        self.set_general_tab_data()
        self.general_layout.addRow("Annotator name: ", self.annotator)
        self.general_layout.addRow("Project Title: ", self.project)
        self.general_layout.addRow("Behaviors: ", self.behaviors)
        
        
        # self.general_layout.addRow("Data type: ", self.data_type_combo)
  
    def set_general_tab_data(self):

        self.annotator = self.set_le("annotator", set_int=False)
        self.project = self.set_le("project", set_int=False)       
        self.behaviors = self.set_multiple_input("actions", type="category")
        
        

    def create_fp_tab(self):
        self.fp_tab = QWidget()
        self.tabs.addTab(self.fp_tab, "File")
        self.fp_layout = QFormLayout()
        self.fp_tab.setLayout(self.fp_layout)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def collect_general(self):
        self.settings["annotator"] = self.annotator.text()
        self.settings["project"] = self.project.text()  
        self.settings["actions"] = (
            self.behaviors.values() if len(self.behaviors.values()) > 0 else None
        )
      

    def create_folder(self) -> None: 
        folder_name = self.settings["project"]
        self.folder_name = self.settings["project"]   
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, folder_name)
        subfolder_names = ["Annotations", "Project Config", "Tracking data"]
        print(f"Folder '{folder_name}' created in '{current_directory}'")
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_path)   
            # Close the dialog after creating the folder
            for subfolder_name in subfolder_names:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                os.makedirs(subfolder_path)  
                
            # Create settings.yaml with default values
            settings_file_path = os.path.join(folder_path, "Project Config", "settings.yaml")
            default_settings = {
                "setting1": "default_value1",
                "setting2": "default_value2",
                # Add more default settings as needed
            }
            with open(settings_file_path, "w") as settings_file:
                yaml.dump(default_settings, settings_file)  
            
            # Read data from default_config.yaml
            default_config_path = os.path.join(current_directory, "default_config.yaml")
            with open(default_config_path, "r") as default_config_file:
                default_config_data = yaml.safe_load(default_config_file)

            # Create config.yaml with default values from default_config.yaml
            config_file_path = os.path.join(folder_path, "Project Config", "config.yaml")
            with open(config_file_path, "w") as config_file:
                yaml.dump(default_config_data, config_file)

            # Get user-defined labels 
            user_labels = self.behaviors.values()

            # Save user-defined labels to annotations.npy in Annotations folder
            annotations_folder_path = os.path.join(folder_path, "Annotations")
            annotations_file_path = os.path.join(annotations_folder_path, "annotations.npy")
            np.save(annotations_file_path, user_labels)

        else:
            print("Folder already exist")
            pass
        
    def close_(self):
        self.close()
    
    def reject(self):
        self.close()
    def getFolderName(self):
        return self.folder_name
    
    def getVideos(self):
        return self.videos
    
    def getSkeletonData(self):
        return self.skeleton
    
    def create_symbolic_link(self, videos, folder_name, current_directory):
        tracking_data_folder_path = os.path.join(current_directory, folder_name)
        
        if not os.path.exists(tracking_data_folder_path):
            os.makedirs(tracking_data_folder_path)
            
        for video in videos:
            video_name = os.path.basename(video)
            link_path = os.path.join(tracking_data_folder_path, video_name)
            try:
                os.symlink(video, link_path)
                print(f"Symbolic link created: {link_path}")
            except Exception as e:
                print(f"Error creating symbolic link for {video_name}: {e}")
    
    def accept(self) -> None:
        self.collect()
        for key, value in list(self.settings.items()):
            if value == "None":
                self.settings[key] = None
        with open(self.config_path, "w") as f:
            YAML().dump(self.settings, f)
        if self.settings["suffix"] is None:
            msg = QMessageBox()
            msg.setText(
                "The annotation suffix parameter cannot be None, please set it to a string value!"
            )
            msg.exec_()
            return
        
        self.create_folder()
        current_directory = os.getcwd()
        folder = self.folder_name
        current_directory = os.path.join(current_directory, folder )
        self.copy_skeleton(self.skeleton, current_directory)
        
        if self.video_checkbox.isChecked():
            print("Copying data")
            self.copy_videos_to_tracking_data(self.videos, "Tracking data", current_directory)
        else: 
            self.create_symbolic_link(self.videos, "Tracking data", current_directory)
            print("Creating a link")

        super().accept()
        self.close()

    def _open_yaml(self, path: str):
        """
        Load a parameter dictionary from a .yaml file
        """

        with open(path) as f:
            data = YAML().load(f)
        if data is None:
            data = {}
        return data
