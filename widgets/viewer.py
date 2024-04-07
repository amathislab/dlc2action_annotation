#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import os
import pickle
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from random import sample as smp
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from PyQt5.Qt import pyqtSignal
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from utils import BoxLoader, Segmentation, get_2d_files, read_skeleton

from .actionbar import Bar
from .canvas import VideoCanvas
from .console import Console
from .core.annotations import save_annotations
from .dialog import (
    AssessmentDialog,
    CatDialog,
    ChoiceDialog,
    ChoiceDialogExample,
    LoadDialog,
)
from .progressbar import ProgressBar
from .sampler import Sampler
from .viewbox import VideoViewBox


class Viewer(QWidget):
    status = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    next_video = pyqtSignal()
    prev_video = pyqtSignal()

    def __init__(
        self,
        stacks,
        shapes,
        lengths,
        output_file,
        labels,
        suggestions,
        settings,
        sequential,
        filenames,
        filepaths,
        current,
        al_mode,
        al_points=None,
    ):
        super(Viewer, self).__init__()
        self.settings = settings
        # Filespaths: Path to video files 
        self.filepaths = filepaths
        self.filenames = filenames
        self.layout = QHBoxLayout()

        self.minus_mode = False
        self.plus_mode = False
        self.al_mode = al_mode
        self.correct_mode = False
        self.annotate_anyway = False

        self.loaded_labels = []
        self.loaded_shortcuts = defaultdict(lambda: {})
        self.loaded_times = None
        self.message = ""
        self.action_dict = settings["actions"]
        self.al_buffer = settings["al_buffer"]
        self.backend = settings["backend"]
        self.al_points = al_points
        self.displayed_animals = []
        self.sequential = sequential
        self.animals = None
        
        cwd = os.getcwd()
        if not cwd.endswith('/Project_Config'):
            os.chdir(os.path.join(os.getcwd(),'Project_Config'))
            with open("colors.txt") as f:
                self.animal_colors = [
                    list(map(lambda x: float(x) / 255, line.split()))
                    for line in f.readlines()
                ]
            os.chdir(cwd)
        else:
            with open("colors.txt") as f:
                self.animal_colors = [
                    list(map(lambda x: float(x) / 255, line.split()))
                    for line in f.readlines()
                ]
                
        self.active = True
        self.display_categories = True
        if self.display_categories:
            self.active_list = "categories"
        else:
            self.active_list = "base"
        self.output_file = output_file
        self.labels_file = labels
        self.suggestions_file = suggestions
        self.n_ind = self.settings["n_ind"]

        if self.al_mode:
            self.al_mode = True
            current, end, current_animal = self.al_points[0]
            current = max(current - self.al_buffer, 0)
            self.al_current = current
            self.al_end = end
            self.cur_al_point = 0
        else:
            self.al_mode = False
            self.al_current = None
            self.al_end = None
            self.cur_al_point = None
            current_animal = None

        self.al_animal = current_animal
        points_df_list, index_dict = self.load_skeleton(current)

        filename = self.filenames[0]
        filepath = self.filepaths[0]
        if self.settings["prefix_separator"] is not None:
            sep = self.settings["prefix_separator"]
            split = filename.split(sep)
            if len(split) > 1:
                filename = sep.join(split[1:])
        self.basename = os.path.join(filepath, filename.split(".")[0])

        if self.labels_file is None and self.settings["suffix"] is not None:
            self.labels_file = self.basename + self.settings["suffix"]
            if not os.path.exists(self.labels_file):
                print(f"{self.labels_file} does not exist")
                self.labels_file = None

        if self.suggestions_file is None:
            self.suggestions_file = self.basename + "_suggestion.pickle"
            if self.suggestions_file == self.labels_file:
                self.suggestions_file = None
            elif not os.path.exists(self.suggestions_file):
                print(f"{self.suggestions_file} does not exist")
                self.suggestions_file = None

        if self.output_file is None:
            if self.labels_file is not None:
                self.output_file = self.labels_file
            elif self.settings["suffix"] is not None:
                self.output_file = self.basename + self.settings["suffix"]
            else:
                raise ValueError("Please set the annotation suffix in settings!")
        self.load_labels()

        data_2d = [None for _ in self.filenames]
        data_3d = None
        if self.settings["3d_suffix"] is not None:
            file_3d = (
                self.output_file.split(".")[0][: -len(self.settings["suffix"])]
                + self.settings["3d_suffix"]
            )
            if os.path.exists(file_3d):
                data_3d = np.load(file_3d)
                if self.settings["calibration_path"] is not None:
                    data_2d = get_2d_files(
                        self.filenames, data_3d, self.settings["calibration_path"]
                    )

        if (
            self.settings["upload_window"]
            and self.settings["skeleton_files"][0] is None
            and self.settings["detection_files"][0] is None
        ):
            self.load_detection()

        if self.settings["detection_files"][0] is not None:
            boxes = []
            for box_file in self.settings["detection_files"]:
                box_loader = BoxLoader(box_file)
                boxes.append(box_loader.get_boxes())
            self.n_ind = box_loader.get_n_ind()
        else:
            boxes = [None for x in self.settings["detection_files"]]

        segmentation_list = self.load_segmentation()
        self.draw_segmentation = False
        for x in segmentation_list:
            if x is not None:
                self.draw_segmentation = True

        self.canvas = VideoCanvas(
            self,
            stacks,
            shapes,
            lengths,
            self.n_ind,
            self.animals,
            boxes,
            points_df_list,
            segmentation_list,
            index_dict,
            current,
            current_animal,
            self.correct_mode,
            self.al_points,
            self.settings["mask_opacity"],
            data_3d,
            data_2d,
            self.settings["skeleton"],
            self.settings["3d_bodyparts"],
        )

        # success = False

        # if type(self.action_dict) is dict or settings["cat_choice"]:
            # Load or create a project window

        
   




            # self.choose_cats()
        self.initialize_cats()

        self.correct_animal = self.current_animal_name() in self.displayed_animals
        self.bar = Bar(
            self,
            self.segment_len(),
            self.al_mode,
            self.al_current,
            self.al_end,
            self.correct_animal,
            None,
            QFontMetrics(self.font()),
        )
        self.bar.setFixedHeight(self.bar.h)
        self.bar.clicked.connect(self.on_bar_clicked)
        self.progressbar = ProgressBar(
            self, self.current, self.loaded, self.video_len()
        )
        self.progressbar.clicked.connect(self.on_bar_clicked)
        self.video_layout = QVBoxLayout()
        self.video_layout.addWidget(self.canvas.native)
        self.video_layout.addWidget(self.bar)
        self.video_layout.addWidget(self.progressbar)
        self.console = Console(self)
        self.console.setMaximumWidth(300)
        self.layout.addLayout(self.video_layout)
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)

        self.canvas.emitter.animal_changed.connect(self.bar.stop_growing)
        self.canvas.emitter.hovered.connect(self.status.emit)
        self.canvas.emitter.point_released.connect(self.emit_message)
        self.canvas.emitter.mode_changed.connect(self.finish_setting)

        self.timer = QTimer()
        self.timer.timeout.connect(self.redraw)
        self.timer.start(self.settings["default_frequency"])

        self.set_move_mode()
        if self.al_mode:
            self.set_al_point()
        else:
            self.set_current(current, center=True)

    def load_segmentation(self):
        if self.settings["segmentation_suffix"] is None:
            segmentation_list = [None for _ in self.filenames]
            return segmentation_list

        self.segmentation_files = [
            os.path.join(fp, fn.split(".")[0] + self.settings["segmentation_suffix"])
            for fp, fn in zip(self.filepaths, self.filenames)
        ]

        segmentation_list = []
        for file in self.segmentation_files:
            if os.path.exists(file):
                segmentation_list.append(Segmentation(file))
            else:
                segmentation_list.append(None)

        return segmentation_list

    def emit_message(self):
        self.status.emit(self.message)

    def change_al_mode(self, value):
        if value and self.al_points is None:
            return
        self.canvas.change_al_mode(value)

    def finish_setting(self, value):
        if not value:
            self.al_mode = False
            self.cur_al_point = None
            self.al_end = None
            self.al_animal = None
            self.al_current = None
            self.console.prev_button.setVisible(False)
            self.console.next_button.setVisible(False)
            self.set_current(self.current())
        else:
            self.al_mode = True
            self.cur_al_point = 0
            self.console.set_buttons(
                ass=self.sampler.assessment(), method=self.sampler.get_method()
            )
            self.set_al_point()
        self.bar.al_mode = value
        self.bar.set_al_point(self.al_current, self.al_end)

    def find_skeleton_files(self):
        
        if len(self.settings["detection_files"]) < len(self.filepaths):
            for i in range(len(self.filepaths) - len(self.settings["detection_files"])):
                self.settings["detection_files"].append(None)
        if self.settings["DLC_suffix"][0] is not None:
            for i in range(len(self.filepaths)):
                if (
                    len(self.settings["skeleton_files"]) <= i
                    or self.settings["skeleton_files"][i] is None
                ):
                    path = None
                    for suffix in self.settings["DLC_suffix"]:
                        possible_path = os.path.join(
                            self.filepaths[i], self.filenames[i].split(".")[0] + suffix
                        )
                        if os.path.exists(possible_path):
                            path = possible_path
                        else:
                            print(f"{possible_path} does not exist")
                    if len(self.settings["skeleton_files"]) <= i:
                        self.settings["skeleton_files"].append(path)
                    else:
                        self.settings["skeleton_files"][i] = path

    def load_skeleton(self, current=None):
        self.find_skeleton_files()
        points_df_list = []
        for skeleton_file in self.settings["skeleton_files"]:
            points_df, index_dict, animals = self.load_animals(skeleton_file)
            points_df_list.append(points_df)

        if current is None:
            current = self.current()

        if index_dict[current] is not None:
            self.displayed_animals = index_dict[current]
        else:
            self.displayed_animals = animals

        self.animals = animals
        self.n_ind = len(self.animals)
        return points_df_list, index_dict

    def choose_cats(self):
        dlg = ChoiceDialog(self.action_dict, self.filenames[0])
        actions, display_categories = dlg.exec_()
        if actions is not None:
            self.settings["actions"] = actions
            self.set_display_categories(display_categories)

    def load_cats(self):
        self.save(verbose=False)
        self.labels_file = self.output_file
        self.load_labels()
        dlg = ChoiceDialog(self.action_dict, self.filenames[0])
        actions, display_categories = dlg.exec_()
        if actions is not None:
            self.settings["actions"] = actions
            self.set_display_categories(display_categories)
            self.initialize_cats()
            self.bar.reset()
            self.get_cats()

    def load_detection(self):
        # TODO: improve logic for file loading, esp. in sequential mode
        dlg = LoadDialog(self.filenames[0])
        boxes, skeletons, labels = dlg.exec_()
        if boxes is not None:
            self.settings["detection_files"] = boxes
        if skeletons is not None:
            self.settings["skeleton_files"] = skeletons
        if labels is not None:
            self.labels_file = labels

    def redraw(self):
        if self.active:
            self.bar.update()
            self.progressbar.update()
            # self.canvas.update()
            # self.bar.update()

    def on_play(self, value=None):
        self.canvas.set_play(value)

    def update_speed(self):
        value = self.console.speed_slider.value()
        self.canvas.change_speed(self.settings["max_frequency"] - value)

    def next(self):
        self.canvas.set_current_frame(self.current() + 1)

    def prev(self):
        self.canvas.set_current_frame(self.current() - 1)

    def set_skeleton_size(self, value):
        self.skeleton_size = value
        for vb in self.canvas.viewboxes:
            vb.set_size(value)
        self.update()

    def set_cut_mode(self):
        self.mode_message = "click an action to split it in two"
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = "C"
        self.bar.mode = "C"
        self.mode_changed.emit("C")

    def set_remove_mode(self):
        self.mode_message = "click an action to remove it"
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = "R"
        self.bar.mode = "R"
        self.mode_changed.emit("R")

    def set_move_mode(self):
        self.mode_message = "move the edges of the actions to change their position"
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = None
        self.bar.mode = None
        self.mode_changed.emit("M")

    def set_new_mode(self):
        if len(self.catDict[self.active_list]) == 0:
            self.show_warning(
                "No categories defined yet! Press 'Change labels' to define."
            )
            return
        self.mode_message = f'click and drag on the action bar to create a new {self.catDict["base"][self.bar.cat]} action'
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = "N"
        self.bar.mode = "N"
        self.mode_changed.emit("N")

    def set_amb_mode(self):
        self.mode_message = "click an action to set it as (not) ambiguous"
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = "A"
        self.bar.mode = "A"
        self.mode_changed.emit("A")

    def set_ass_mode(self):
        if len(self.catDict[self.active_list]) == 0:
            self.show_warning(
                "No categories defined yet! Press 'Change labels' to define."
            )
            return
        self.mode_message = f'click on an action to assign the {self.catDict["base"][self.bar.cat]} label'
        self.message = self.mode_message
        self.status.emit(self.mode_message)
        self.mode = "As"
        self.bar.mode = "As"
        self.mode_changed.emit("As")

    def set_display_names(self, state):
        self.canvas.set_display_names(state)

    def item_clicked(self, event):
        if self.mode != "As":
            self.set_new_mode()
        self.set_cat(event)

    def set_cat(self, event):
        if type(event) is str:
            cat_name = event
        else:
            try:
                cat_name = event.text().split(" (")[0]
            except:
                return
        self.bar.set_cat(cat_name)

    def set_box_freq(self, value):
        value = self.settings["detection_update_freq"] + 4 - value
        if value >= self.settings["detection_update_freq"] + 3:
            self.canvas.hide_boxes()
        else:
            self.canvas.set_box_update(value)


    def initialize_cats(self):
        self.shortCut = defaultdict(lambda: {})
        self.catDict = defaultdict(lambda: [])
        self.invisible_actions = []
        taken = defaultdict(lambda: [str(i) for i in range(min(self.n_ind, 10))])
        actions = []
        for key in self.settings["actions"]:
            actions += self.settings["actions"][key]
            actions += [key]
            self.catDict[key] = [
                x for x in self.settings["actions"][key] if not x.startswith("negative")
            ]
            self.catDict["categories"].append(key)
        for a in self.loaded_labels:
            if a not in actions:
                actions.append(a)
                self.invisible_actions.append(a)
        self.catDict["base"] = {}
        self.neg_actions = [x for x in actions if x.startswith("negative")]
        self.unknown_actions = [x for x in actions if x.startswith("unknown")]
        actions = [
            x
            for x in actions
            if not x.startswith("negative") and not x.startswith("unknown")
        ]
        self.times = [[np.array([]) for j in actions] for i in range(self.n_animals())]
        self.negative_times = [
            [np.array([]) for j in self.neg_actions] for i in range(self.n_animals())
        ]
        self.unknown_times = [
            [np.array([]) for j in self.unknown_actions]
            for i in range(self.n_animals())
        ]
        for i, a in enumerate(actions):
            self.catDict["base"][i] = a
            if a in self.loaded_labels:
                a_i = self.loaded_labels.index(a)
                for ind, name in enumerate(self.animals):
                    if name in self.loaded_animals:
                        ind_i = self.loaded_animals.index(name)
                        self.times[ind][i] = self.loaded_times[ind_i][a_i]
                    else:
                        self.times[ind][i] = []
        for i, a in enumerate(self.neg_actions):
            if a in self.loaded_labels:
                a_i = self.loaded_labels.index(a)
                for ind, name in enumerate(self.animals):
                    if name in self.loaded_animals:
                        ind_i = self.loaded_animals.index(name)
                        self.negative_times[ind][i] = self.loaded_times[ind_i][a_i]
                    else:
                        self.negative_times[ind][i] = []
        for i, a in enumerate(self.unknown_actions):
            if a in self.loaded_labels:
                a_i = self.loaded_labels.index(a)
                for ind, name in enumerate(self.animals):
                    if name in self.loaded_animals:
                        ind_i = self.loaded_animals.index(name)
                        self.unknown_times[ind][i] = self.loaded_times[ind_i][a_i]
                    else:
                        self.unknown_times[ind][i] = []
        cat_dict = defaultdict(lambda: {})
        for a_i, a in self.catDict["base"].items():
            if a in self.catDict["categories"]:
                cat_dict["categories"][a_i] = a
            for category in self.catDict["categories"]:
                if a in self.catDict[category]:
                    cat_dict[category][a_i] = a
        for k, v in cat_dict.items():
            self.catDict[k] = v
        for k, v in self.catDict.items():
            for i, a in v.items():
                if a not in self.invisible_actions:
                    if a in self.loaded_shortcuts[k].keys():
                        sc = self.loaded_shortcuts[k][a]
                        self.shortCut[k][sc] = i
                        taken[k].append(sc)
        for k, v in self.catDict.items():
            for i, a in v.items():
                if a not in self.invisible_actions and a[0].upper() not in taken[k]:
                    self.shortCut[k][a[0].upper()] = i
                    taken[k].append(a[0].upper())
        self.ncat = len(self.catDict["base"])
        self.get_ncat()
        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        sugg_labels = set()
        for ind_list in self.times:
            for i, cat in enumerate(cat_labels):
                if (
                    cat not in sugg_labels
                    and len(ind_list[i]) > 0
                    and len([x for x in ind_list[i] if x[-1] > 1]) > 0
                ):
                    sugg_labels.add(cat)
        self.sampler = Sampler(classes=list(sugg_labels))

    def get_ncat(self):
        self.ncat = len(self.catDict["base"])
        try:
            ncat_old = len(self.times[0])
            if self.ncat > ncat_old:
                for ind_list in self.times:
                    ind_list += [[] for i in range(self.ncat - ncat_old)]
        except:
            pass

    # FUNCTION TO HANDLE "CHANGE LABELS" ------------------
    def get_cats(self):
        # When you change labels you want to have all your labels listed
        # if self.display_categories:
        #     dialog = CatDialog(
        #         self.catDict, self.shortCut, self.invisible_actions, "categories", self
        #     )
        #     self.catDict, self.shortCut, self.invisible_actions, _ = dialog.exec_()
            # keys = [key for key in self.catDict if key not in ["base", "categories"]]
        
        # else:
        #     keys = ["base"]
        # for key in keys:
        
        # Changed KEY to "BASE"
            dialog = CatDialog(
                self.catDict, self.shortCut, self.invisible_actions, "base", self
            )
            (
                self.catDict,
                self.shortCut,
                self.invisible_actions,
                actions,
            ) = dialog.exec_()
        #     if key != "base":
        #         self.settings["actions"][key] = actions
        #     else:
        #         for action in actions:
        #             success = False
        #             for category, label_dict in self.catDict.items():
        #                 label_list = [v for _, v in label_dict.items()]
        #                 if action in label_list and category != "base":
        #                     success = True
        #                     if category == "categories":
        #                         if action not in self.settings["actions"]:
        #                             self.settings["actions"][action] = []
        #                     else:
        #                         if category not in self.settings["actions"]:
        #                             self.settings["actions"][category] = []
        #                         if action not in self.settings["actions"][category]:
        #                             self.settings["actions"][category].append(action)
        #                     break
        #             if not success:
        #                 if "other" not in self.settings["actions"]:
        #                     self.settings["actions"]["other"] = []
        #                 if action not in self.settings["actions"]["other"]:
        #                     self.settings["actions"]["other"].append(action)

        # self.ncat = len(self.catDict["base"])
        # self.get_ncat()
        # try:
        #     self.console.catlist.update_list()
        #     self.update_animals()
        #     self.console.seglist.update_list()
        #     self.update()
        # except:
        #     pass

    def current(self):
        return self.canvas.current

    def loaded(self):
        if self.al_mode:
            loaded = self.canvas.loaded[self.cur_al_point]
        else:
            loaded = self.canvas.loaded
        return loaded

    def set_current(self, value, center=False):
        self.canvas.set_current_frame(value, center)

    def video_len(self):
        return self.canvas.len_global

    def segment_len(self):
        return 200

    def n_animals(self):
        return self.canvas.n_ind

    def animal(self, x):
        return self.canvas.animals[x]

    def current_animal(self):
        cur = self.canvas.current_animal
        if cur not in self.canvas.animals:
            return 0
        else:
            return self.canvas.animals.index(cur)

    def current_animal_name(self):
        cur = self.current_animal()
        return self.animals[cur]

    def set_edges(self, center=False):
        l = self.segment_len()
        cur = self.current()
        if not self.canvas.play and not center:
            if cur > self.bar.start and cur < self.bar.start + l:
                return
        if center:
            s = cur - l // 2
        else:
            s = self.bar.start
            m = s + l // 2
            if cur > m + 1:
                s += 2
            elif cur == m + 1:
                s += 1
        if cur - l // 2 < 0:
            s = 0
        elif cur + l // 2 > self.video_len():
            s = self.video_len() - l
        self.bar.start = s

    def shortCutInv(self, key="base"):
        inv = {v: k for k, v in self.shortCut[key].items()}
        return inv

    def catDictInv(self, key="base"):
        inv = {v: k for k, v in self.catDict[key].items()}
        return inv

    def export_annotation_data(self) -> Tuple[dict, list, list, list]:
        """
        Updates the annotation times by calling self.update_labels(), and then prepares
        the annotation data to be exported

        Returns:
            metadata
            category_labels
            animals
            times
        """
        self.update_labels()
        times = deepcopy(self.times)
        for i, ind_list in enumerate(times):
            for j, cat_list in enumerate(ind_list):
                to_remove = []
                for k, label in enumerate(cat_list):
                    if label[-1] in [2, 3]:
                        to_remove.append(k)
                times[i][j] = np.delete(times[i][j], to_remove, 0)

        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        neg_classes = self.settings["hard_negative_classes"]
        if neg_classes == "all":
            neg_classes = [x for x in cat_labels if x not in self.invisible_actions]

        if neg_classes is not None and self.al_points is not None:
            for neg_class in neg_classes:
                neg_ind = cat_labels.index(neg_class)
                cat_labels.append(f"negative {neg_class}")
                for i, ind_list in enumerate(times):
                    labels = np.zeros(self.video_len())
                    for start, end, ind in self.al_points:
                        if ind == self.animals[i]:
                            labels[start:end] = -1
                    if f"negative {neg_class}" in self.neg_actions:
                        neg_a_i = self.neg_actions.index(f"negative {neg_class}")
                        for start, end, _ in self.negative_times[i][neg_a_i]:
                            labels[start:end] = -1
                    for start, end, _ in times[i][neg_ind]:
                        labels[start:end] = 1
                    y_ext = np.r_[False, labels == -1, False]
                    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])
                    times[i].append([])
                    for start, end in zip(idx[:-1:2], idx[1::2]):
                        times[i][-1].append([start, end, 0])

        for j, a in enumerate(self.neg_actions):
            if self.neg_actions[j] not in cat_labels:
                cat_labels.append(self.neg_actions[j])
                for i in range(len(times)):
                    times[i].append(self.negative_times[i][j])

        for j, a in enumerate(self.unknown_actions):
            if self.unknown_actions[j] not in cat_labels:
                cat_labels.append(self.unknown_actions[j])
                for i in range(len(times)):
                    times[i].append(self.unknown_times[i][j])

        metadata = {
            "datetime": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "annotator": self.settings["annotator"],
            "length": self.video_len(),
            "video_file": self.filenames[0],
            "skeleton_files": self.settings["skeleton_files"],
        }
        return metadata, cat_labels, self.animals, times
    
  
    def save(self, event=None, verbose=True, new_file=False, ask=False):
        if ask:
            msg = QMessageBox()
            msg.setText("Save the current annotations?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.No:
                return

        metadata, cat_labels, animals, times = self.export_annotation_data()
        if self.output_file is None or new_file:
            self.output_file = QFileDialog.getSaveFileName(self, "Save file")[0]
            if len(self.output_file) == 0:
                self.output_file = None
                return False

        # TODO: WHY IS THE LAST ACTION CHOICE SAVED HERE
        with open("../last_action_choice.pickle", "wb") as f:
            loaded_shortcuts = defaultdict(lambda: {})
            for k in self.catDict:
                for i, sc in self.shortCutInv(k).items():
                    cat = self.catDict[k][i]
                    loaded_shortcuts[k][cat] = sc
            pickle.dump(
                (
                    self.settings["actions"],
                    self.display_categories,
                    dict(loaded_shortcuts),
                ),
                f,
            )

        # TODO: SAVING
        try:
            save_annotations(
                output_path=Path(self.output_file),
                metadata=metadata,
                animals=animals,
                cat_labels=cat_labels,
                times=times,
                human_readable=True,
                overwrite=True,
            )
            
            # if verbose:
            #     self.show_warning("Saved successfully!")
            
        except IOError as err:
            warnings.warn(f"Failed to save annotation data: {err}")

        return True

    def show_warning(self, message):
        msg = QMessageBox()
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_question(self, message, default):
        self.on_play(False)
        msg = QMessageBox()
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
        if default == "yes":
            msg.setDefaultButton(QMessageBox.Yes)
        elif default == "no":
            msg.setDefaultButton(QMessageBox.No)
        reply = msg.exec_()
        if reply == QMessageBox.No:
            return False
        else:
            return True

    def load_prior(self, file, amb_replace):
        with open(file, "rb") as f:
            _, loaded_labels, animals, loaded_times = pickle.load(f)
        animals_i = []
        for ind in animals:
            if ind in self.loaded_animals:
                animals_i.append(self.loaded_animals.index(ind))
            else:
                animals_i.append(None)
        labels_i = []
        for cat in loaded_labels:
            if cat in self.loaded_labels:
                labels_i.append(self.loaded_labels.index(cat))
            else:
                labels_i.append(None)
        for i, ind_list in zip(animals_i, loaded_times):
            if i is not None:
                for j, cat_list in zip(labels_i, ind_list):
                    if j is not None:
                        cat_list = list(cat_list)
                        if len(self.loaded_times[i][j]) > 0:
                            self.loaded_times[i][j] = np.array(self.loaded_times[i][j])
                            to_remove = []
                            # for k, (start, end, amb) in enumerate(cat_list):
                            #     mask = (self.loaded_times[i][j][:, 0] < end) & (
                            #         self.loaded_times[i][j][:, 1] >= start
                            #     )
                            #     if np.sum(mask) > 0:
                            #         to_remove.append(cat_list[k])
                            # for element in to_remove:
                            #     cat_list = [
                            #         x
                            #         for x in cat_list
                            #         if not np.array_equal(x, element)
                            #     ]
                        self.loaded_times[i][j] = np.array(
                            list(self.loaded_times[i][j]) + cat_list
                        )

    def load_labels(self):
        try:
            try:
                self.load_annotation(self.labels_file)
            except:
                with pd.HDFStore(self.labels_file) as store:
                    self.loaded_data = store["annotation"]
                    try:
                        metadata = store.get_storer("annotation").attrs.metadata
                        print(
                            f'loaded labels annotated by {metadata["annotator"]} at {metadata["datetime"]} for {metadata["video_file"]}'
                        )
                    except:
                        print("loaded labels")
                self.n_ind = len(self.loaded_data.columns.unique("individuals"))
                self.loaded_labels = list(self.loaded_data.columns.unique("categories"))
                self.animals = list(self.loaded_data.columns.unique("individuals"))
                n_cat = len(self.loaded_data.columns.unique("categories"))
                self.loaded_data = self.loaded_data.to_numpy().T.reshape(
                    (self.n_ind, n_cat, -1)
                )
                times = [[] for i in range(self.n_ind)]
                for ind_i, ind_list in enumerate(self.loaded_data):
                    for i in range(ind_list.shape[0]):
                        l = copy(ind_list[i, :])
                        l = (l == 0.5).astype(int)
                        list_amb = [
                            [*x, True]
                            for x in np.flatnonzero(
                                np.diff(np.r_[0, l, 0]) != 0
                            ).reshape(-1, 2)
                        ]
                        l = copy(ind_list[i, :])
                        l = (l == 1).astype(int)
                        list_sure = [
                            [*x, False]
                            for x in np.flatnonzero(
                                np.diff(np.r_[0, l, 0]) != 0
                            ).reshape(-1, 2)
                        ]
                        times[ind_i].append(np.array(list_amb + list_sure))
                self.loaded_times = times
                del self.loaded_data

        except:
            if self.labels_file is not None and os.path.exists(self.labels_file):
                print(f"annotation file {self.labels_file} is invalid")
            elif self.labels_file is not None:
                print(f"annotation file {self.labels_file} does not exist")

        if self.suggestions_file is not None:
            if self.loaded_times is None:
                self.load_annotation(self.suggestions_file, 2)
            else:
                self.load_prior(self.suggestions_file, 2)

        # priors = self.settings["prior_suffix"]
        # if priors is None:
        #     priors = []
        # elif type(priors) is not list:
        #     priors = [priors]
        # priors = priors + ["_suggestion"]
        # for prior in priors:
        #     if prior == "_suggestion":
        #         amb = 2
        #     else:
        #         amb = 3
        #     file = (
        #         self.output_file.split(".")[0][: -len(self.settings["suffix"])]
        #         + prior
        #         + ".pickle"
        #     )
        #     if os.path.exists(file):
        #         if self.loaded_times is None:
        #             self.load_annotation(file, amb)
        #         else:
        #             self.load_prior(file, amb)

    def load_annotation(self, file, amb=None):
        if self.settings["data_type"] == "dlc":
            with open(file, "rb") as f:
                (
                    metadata,
                    self.loaded_labels,
                    self.loaded_animals,
                    self.loaded_times,
                ) = pickle.load(f)

            if self.animals is None:
                self.animals = self.loaded_animals
            else:
                self.anmals = list(set(self.animals + self.loaded_animals))
            self.n_ind = len(self.animals)
            if amb is not None:
                for i, ind_list in enumerate(self.loaded_times):
                    for j, cat_list in enumerate(ind_list):
                        for k in range(len(cat_list)):
                            self.loaded_times[i][j][k][-1] = amb
        elif self.settings["data_type"] == "calms21":
            f = np.load(file, allow_pickle=True).item()
            keys = sorted(list(f.keys()))
            seq = keys[0]
            f = f[seq]
            self.loaded_animals = ["ind0", "ind1"]
            self.loaded_labels = ["attack", "investigation", "mount"]
            self.loaded_times = [[[] for i in range(3)] for i in range(2)]
            from itertools import groupby

            repeated = [(k, sum(1 for i in g)) for k, g in groupby(f)]
            cur = 0
            for label, length in repeated:
                if label != 3:
                    self.loaded_times[0][label].append([cur, cur + length, False])
                cur += length

    def update_labels(self):
        self.bar.set_data_from_rects(final=True)
        self.times[self.current_animal()] = self.bar.times

    def set_animal(self, event):
        if event is None:
            return
        try:
            animal = event.text().split()[0]
        except:
            if type(event) is int:
                self.console.animallist.selected = event
                animal = self.animal(event)
            elif type(event) is str:
                animal = event
        self.update_labels()
        if animal not in self.displayed_animals:
            self.set_correct_animal(False)
        self.canvas.set_current_animal(animal)
        for vb in self.canvas.viewboxes:
            if isinstance(vb, VideoViewBox) and vb.points_df is not None:
                vb.draw_points(
                    self.current(),
                    self.displayed_animals,
                    self.skeleton_color,
                    self.al_mode,
                    self.al_animal,
                )
        self.bar.get_labels()
        self.update_animals()

    def change_tracklet(self):
        if self.annotate_anyway:
            ok = True
        else:
            ok = self.show_question(
                message="This tracklet has ended. Annotate anyway?", default="no"
            )
        if not ok:
            self.on_play(False)
            self.bar.stop_growing()
        else:
            self.annotate_anyway = True
            self.on_play(True)

    def set_correct_animal(self, value):
        self.correct_animal = value
        self.bar.correct_animal = value

    def skeleton_color(self, i):
        return self.animal_colors[int(i % len(self.animal_colors))]

    def get_animals(self):
        return [
            (self.animal(i), self.skeleton_color(i)) for i in range(self.n_animals())
        ]

    def get_displayed_animals(self):
        try:
            return [
                (ind, self.skeleton_color(self.animals.index(ind)))
                for ind in self.displayed_animals
            ]
        except:
            raise ValueError("The individual lists in uploaded files are different")

    def load_animals(self, skeleton_file):
        if skeleton_file is not None:
            # try:
            points_df, index_dict = read_skeleton(
                skeleton_file,
                self.settings["data_type"],
                self.settings["likelihood_cutoff"],
                self.settings["min_length_frames"],
            )
            animals = points_df.animals
            # except:
            #     print(f"skeleton file {skeleton_file} is invalid or does not exist")
            #     points_df = None
            #     animals = [f"ind{i}" for i in range(self.settings["n_ind"])]
            #     index_dict = defaultdict(lambda: None)
        else:
            points_df = None
            animals = [f"ind{i}" for i in range(self.settings["n_ind"])]
            index_dict = defaultdict(lambda: None)
        return points_df, index_dict, animals

    def on_next(self):
        if self.al_mode:
            self.next_al_point()
        else:
            self.next_video_f()

    def on_prev(self):
        if self.al_mode:
            self.prev_al_point()
        else:
            self.prev_video_f()

    def on_edit_next(self):
        self.update_labels()
        old_times = self.al_points[self.cur_al_point][:2]
        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        ind = cat_labels.index(self.sampler.get_behavior())
        new_times = self.bar.get_new_times(ind)
        starts, ends = self.sampler.update_edit(old_times, new_times)
        neg_label = f"negative {self.sampler.get_behavior()}"
        if neg_label not in self.neg_actions:
            self.neg_actions.append(neg_label)
            for i in range(len(self.negative_times)):
                self.negative_times[i].append([])
        neg_ind = self.neg_actions.index(neg_label)
        for s, e in zip(starts, ends):
            self.negative_times[self.current_animal()][neg_ind].append([s, e, 0])
        self.negative_times[self.current_animal()][neg_ind] = sorted(
            self.negative_times[self.current_animal()][neg_ind]
        )
        start, end = old_times
        ind_i = self.current_animal()
        interval = np.ones(end - start)
        for cat_ind in range(len(cat_labels)):
            if cat_ind == ind:
                continue
            for s, e, amb in self.times[ind_i][cat_ind]:
                if s < end and e > start:
                    s = max(0, s - start)
                    e = min(end - start, e - start)
                    if amb < 2:
                        interval[s:e] = 0
        interval = np.array([0] + list(interval) + [0])
        diffs = np.diff(interval)
        starts = list(np.argwhere(diffs == 1) + start)
        ends = list(np.argwhere(diffs == -1) + start)
        if len(starts) > 0:
            for cat_ind, cat in enumerate(cat_labels):
                if cat_ind == ind:
                    continue
                un_label = f"unknown {cat}"
                if un_label not in self.unknown_actions:
                    self.unknown_actions.append(un_label)
                    for i in range(len(self.unknown_times)):
                        self.unknown_times[i].append([])
                un_ind = self.unknown_actions.index(un_label)
                for (
                    s,
                    e,
                ) in zip(starts, ends):
                    self.unknown_times[ind_i][un_ind].append([s, e, 0])
        self.on_next()

    def on_edit_prev(self):
        self.on_z()

    def on_good(self):
        self.sampler.update_good(1)
        self.on_next()

    def on_bad(self):
        self.sampler.update_good(0)
        self.on_next()

    def next_video_f(self):
        
        # if self.show_question(message="Save the current annotation?", default="yes"):
        #     success = self.save()
        # else:
        #     success = True
        # if success:
        
        # Instead of asking users if they want to save the current annotation
        # we save it automatically
        
        self.save()
        self.next_video.emit()

    def prev_video_f(self):
        # if self.show_question(message="Save the current annotation?", default="yes"):
        #     success = self.save()
        # else:
        #     success = True
        # if success:
        self.save()
        self.prev_video.emit()

    def next_al_point(self):
        cur_al_point = self.cur_al_point + 1
        if cur_al_point >= len(self.al_points):
            self.change_al_mode(False)
            if self.sampler.assessment():
                self.sampler.compute()
                self.set_assessment_al()
            elif self.show_question(message="Move on from this video?", default="yes"):
                self.next_video_f()
        else:
            self.cur_al_point = cur_al_point
            self.set_al_point()

    def prev_al_point(self):
        cur_al_point = self.cur_al_point - 1
        if cur_al_point < 0:
            if self.show_question(
                message="Move on to a different video?", default="yes"
            ):
                self.prev_video_f()
        else:
            self.cur_al_point = cur_al_point
            self.set_al_point()

    def set_al_point(self):
        cur, end, animal = self.al_points[self.cur_al_point]
        self.set_current(max(cur - self.al_buffer, 0), center=True)
        self.al_animal = animal
        self.al_current = cur
        self.al_end = end
        self.bar.set_al_point(cur, end)
        self.canvas.set_al_point(al_point=self.cur_al_point)
        self.set_animal(animal)

    def update_animals(self):
        self.console.animallist.update_list(
            self.current_animal_name(), self.get_displayed_animals()
        )

    def change_displayed_animals(self, ind_list):
        if set(self.displayed_animals) != set(ind_list):
            self.displayed_animals = ind_list
            self.update_animals()
            if self.current_animal_name() not in self.displayed_animals:
                self.set_correct_animal(False)
                if len(self.bar.grow_rects) > 0:
                    self.change_tracklet()
            else:
                self.annotate_anyway = False
                self.set_correct_animal(True)

    def set_active_list(self, key):
        self.active_list = key
        self.console.catlist.set_key(key)
        if key == "base":
            self.console.back_button.setVisible(False)
        else:
            self.console.back_button.setVisible(True)
        if key not in ["base", "categories"]:
            self.console.back_button.setEnabled(True)
        else:
            self.console.back_button.setEnabled(False)

    def on_shortcut(self, sc):
        self.bar.on_shortcut(sc)
        if self.active_list == "categories":
            key = self.shortCut["categories"][sc]
            name = self.catDict["base"][key]
            self.set_active_list(name)
            self.status.emit("press Esc to go back to categories")

    def on_escape(self):
        if self.minus_mode:
            self.set_minus_mode(False)
        elif self.active_list not in ["base", "categories"]:
            self.set_active_list("categories")
        self.message = self.mode_message
        self.status.emit(self.mode_message)

    def set_minus_mode(self, value):
        self.minus_mode = value
        if value:
            self.set_plus_mode(False)
            self.bar.set_minus_mode(True)
            self.message = (
                "press Enter to stop the chosen action, - to choose another action, "
                "Esc to quit this mode"
            )
            self.status.emit(self.message)
        else:
            self.bar.set_minus_mode(False)

    def set_plus_mode(self, value):
        self.plus_mode = value
        if value:
            self.set_minus_mode(False)
            self.bar.set_plus_mode(True)
            self.message = (
                "press Enter to set the chosen action as (not) ambiguous, "
                "= to choose another action, Esc to quit this mode"
            )
            self.status.emit(self.message)
        else:
            self.bar.set_plus_mode(False)

    def on_minus(self):
        if not self.minus_mode and len(self.bar.grow_rects) > 0:
            self.set_minus_mode(True)
        elif self.minus_mode:
            self.bar.move_minus()

    def on_plus(self):
        if not self.plus_mode and len(self.bar.grow_rects) > 0:
            self.set_plus_mode(True)
        elif self.plus_mode:
            self.bar.move_plus()

    def on_enter(self):
        if self.minus_mode:
            self.minus_mode = False
            self.bar.stop_minus()
        elif self.plus_mode:
            self.plus_mode = False
            self.bar.stop_plus()
        self.message = self.mode_message
        self.status.emit(self.mode_message)

    def set_display_categories(self, value):
        self.display_categories = value
        if value and self.active_list == "base":
            self.active_list = "categories"
        elif not value:
            self.active_list = "base"
        try:
            self.set_active_list(self.active_list)
        except:
            pass

    def doubleclick(self, item):
        if self.active_list == "categories":
            cat_name = item.text().split(" (")[0]
            self.set_active_list(cat_name)

    def active_shortcuts(self):
        return self.shortCut[self.active_list].keys()

    # SAVE CORRECTION WORKFLOW ------------
    
    # def set_correct_mode(self, event):
    #     self.correct_mode = True
    #     for vb in self.canvas.viewboxes:
    #         vb.correct_mode = True
    #     self.message = 'Click and drag a keypoint to correct a DLC error. Press "Save correction" when you are done'
    #     self.status.emit(self.message)
    #     self.console.correct_button.setVisible(True)

    # def save_correction(self, event):
    #     self.correct_mode = False
    #     for i, vb in enumerate(self.canvas.viewboxes):
    #         if isinstance(vb, VideoViewBox):
    #             vb.correct_mode = False
    #             corrections = vb.get_keypoints(self.current())
    #             if corrections is not None:
    #                 fp = self.filepaths[i]
    #                 fn = (
    #                     os.path.basename(self.filenames[i]).split(".")[0]
    #                     + "_corrections.pickle"
    #                 )
    #                 file = os.path.join(fp, fn)
    #                 if os.path.exists(file):
    #                     with open(file, "rb") as f:
    #                         data = pickle.load(f)
    #                         data.update(corrections)
    #                         corrections = data
    #                 # TODO: DUMPING CORRECTIONS
    #                 with open(file, "wb") as f:
    #                     pickle.dump(corrections, f)
    #                 im = Image.fromarray(vb.get_image(self.current()))
    #                 fn = (
    #                     os.path.basename(self.filenames[i]).split(".")[0]
    #                     + f"_frame{self.current():07}.jpeg"
    #                 )
    #                 file = os.path.join(fp, fn)
    #                 im.save(file)
    #     self.message = self.mode_message
    #     self.status.emit(self.message)
    #     self.console.correct_button.setVisible(False)

    def on_bar_clicked(self, cur):
        self.set_current(cur, center=True)
        self.bar.stop_growing()
        for vb in self.canvas.viewboxes:
            vb.set_image(
                cur,
                displayed_animals=self.displayed_animals,
                skeleton_color=self.skeleton_color,
                al_mode=self.al_mode,
                al_animal=self.al_animal,
            )

    def on_z(self):
        if self.sampler.assessment() and self.sampler.undo():
            self.on_prev()

    def set_unlabeled_al(self):
        self.al_points = []
        for ind_i, ind in enumerate(self.animals):
            labels = np.zeros((len(self.times[ind_i]), self.video_len()))
            for cat in range(len(self.times[ind_i])):
                for start, end, _ in self.times[ind_i][cat]:
                    labels[cat, start:end] = 1
            from itertools import groupby

            cur = 0
            for k, g in groupby(np.sum(labels, axis=0) > 0):
                length = len(list(g))
                if not k and length > 20:
                    self.al_points.append([cur, cur + length, ind])
                cur += length
        self.canvas.al_points = self.al_points

    def set_assessment_al(self):
        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        dlg = AssessmentDialog(self.sampler)
        label = dlg.exec_()
        if label is None:
            return False
        al_points = []
        label_id = cat_labels.index(label)
        for ind_i, ind in enumerate(self.animals):
            for start, end, amb in self.times[ind_i][label_id]:
                if amb > 1:
                    al_points.append([start, end, ind])
        if len(al_points) == 0:
            self.show_warning(f"No intervals found with label {label}!")
            return False
        ass_n = self.settings["assessment_n"]
        if len(al_points) < ass_n:
            ass_n = len(al_points)
        al_points = smp(al_points, k=ass_n)
        self.al_points = al_points
        self.sampler.start_sampling(label)
        self.canvas.al_points = self.al_points
        self.change_al_mode(True)
        return True

    def set_label_al(self):
        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        used_labels = set()
        for ind_list in self.times:
            for i, cat in enumerate(cat_labels):
                if len(ind_list[i]) > 0:
                    used_labels.add(cat)
        label, ok = QInputDialog.getItem(
            self,
            "Choose the label to search for",
            "List of labels",
            sorted(list(used_labels)),
            0,
            False,
        )
        if ok:
            self.al_points = []
            label_id = cat_labels.index(label)
            for ind_i, ind in enumerate(self.animals):
                for start, end, _ in self.times[ind_i][label_id]:
                    self.al_points.append([start, end, ind])
            if len(self.al_points) > 0:
                self.canvas.al_points = self.al_points
                return True
            else:
                self.al_points = None
                return False
        else:
            return False

    def set_tracklet_al(self):
        self.al_points = []
        for ind in self.animals:
            start, end = self.canvas.get_ind_start_end(ind)
            if start is not None and end is not None:
                self.al_points.append([start, end, ind])
        self.canvas.al_points = self.al_points

    def set_cat_id(self, cat_id):
        self.canvas.set_cat_id(cat_id)

    def get_segmentation_cats(self):
        return self.canvas.get_segmentation_cats()

    def set_mask_opacity(self, value):
        self.canvas.set_mask_opacity(value / 10)
        self.update()

    def switch_rainbow(self, value):
        self.canvas.switch_rainbow()

    def switch_skeleton(self, value):
        self.canvas.switch_skeleton()

    def switch_repr(self, value):
        self.canvas.switch_repr()

    def export_examples(self, value):
        from moviepy.editor import VideoFileClip
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

        dlg = ChoiceDialogExample(self.action_dict)
        labels = dlg.exec_()
        cat_labels = [self.catDict["base"][i] for i in self.catDict["base"]]
        num_clips = 5
        target_dir = os.path.join(
            QFileDialog.getExistingDirectory(self, "Save File"), "extracted_clips"
        )
        fps = VideoFileClip(os.path.join(self.filepaths[0], self.filenames[0])).fps
        os.mkdir(target_dir)
        for label in labels:
            os.mkdir(os.path.join(target_dir, label))
            cnt = 0
            if label not in cat_labels:
                continue
            label_id = cat_labels.index(label)
            for ind_i, ind in enumerate(self.animals):
                cat_list = self.times[ind_i][label_id]
                if len(cat_list) == 0:
                    continue
                i = 0
                while cnt < num_clips and i < len(cat_list):
                    start, end, amb = cat_list[i]
                    i += 1
                    if amb != 0 or end - start < 2 * fps // 3:
                        continue
                    for fn, fp in zip(self.filenames, self.filepaths):
                        filename_in = os.path.join(fp, fn)
                        name, ext = fn.split(".")
                        filename_out = os.path.join(
                            target_dir, label, f"{name}_{label}_{cnt}.{ext}"
                        )
                        ffmpeg_extract_subclip(
                            filename_in, start / fps, end / fps, targetname=filename_out
                        )
                        cnt += 1
                        print(f"exported {filename_out}")
            if cnt < num_clips:
                print(f"not enough clips for {label}")
