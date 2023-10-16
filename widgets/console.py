#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
from PyQt5.Qt import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .lists import AnimalList, CatList, SegmentationList


class Console(QWidget):
    def __init__(self, window):
        super(Console, self).__init__()

        self.window = window
        self.layout = QVBoxLayout()
        self.setMinimumWidth(self.window.settings["console_width"])

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(self.window.settings["max_frequency"] - 1)
        self.speed_slider.setValue(
            self.window.settings["max_frequency"]
            - self.window.settings["default_frequency"]
        )
        self.speed_slider.valueChanged.connect(self.window.update_speed)
        self.speed_label = QLabel("Video speed:")

        self.catlist = CatList(key=self.window.active_list, window=self.window)
        self.catlist.itemClicked.connect(self.window.item_clicked)
        self.catlist.itemDoubleClicked.connect(self.window.doubleclick)
        self.animallist = AnimalList(
            window=self.window,
            current=self.window.current_animal_name(),
            visuals=self.window.get_displayed_animals(),
        )
        self.animallist.itemClicked.connect(self.window.set_animal)
        self.seglist = SegmentationList(
            self.window.get_segmentation_cats(), window=self.window
        )
        self.seglist.itemsChecked.connect(self.window.set_cat_id)

        self.indlabel = QLabel("Individuals:")
        self.catlabel = QLabel("Labels:")
        self.seglabel = QLabel("Segmentation:")

        self.box_slider = QSlider(Qt.Horizontal)
        self.box_slider.setMaximum(self.window.settings["detection_update_freq"] + 3)
        self.box_slider.setMinimum(1)
        self.box_slider.setTickPosition(QSlider.TicksBelow)
        self.box_slider.setTickInterval(1)
        self.box_slider.setSingleStep(1)
        self.box_slider.setValue(4)
        self.box_slider.valueChanged.connect(self.window.set_box_freq)
        self.box_label = QLabel("Detection update:")

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMaximum(6)
        self.size_slider.setMinimum(0)
        self.size_slider.setSingleStep(1)
        self.size_slider.setValue(4)
        self.size_slider.valueChanged.connect(self.window.set_skeleton_size)
        self.size_label = QLabel("Marker size:")

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMaximum(10)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setValue(self.window.settings["mask_opacity"] * 10)
        self.opacity_slider.setSingleStep(1)
        self.opacity_slider.valueChanged.connect(self.window.set_mask_opacity)
        self.opacity_label = QLabel("Mask opacity:")

        self.ind_names_tick = QCheckBox()
        self.ind_names_tick.setChecked(False)
        self.ind_names_tick.stateChanged.connect(self.window.set_display_names)
        self.ind_names_label = QLabel("Display names:")

        # self.correct_button = QPushButton("Save correction")
        self.correct_button.clicked.connect(self.window.save_correction)
        self.correct_button.setVisible(self.window.correct_mode)

        self.speed_form = QFormLayout()
        self.speed_form.addRow(self.speed_label, self.speed_slider)
        if any([x is not None for x in self.window.settings["detection_files"]]) or any(
            [x is not None for x in self.window.settings["skeleton_files"]]
        ):
            self.speed_form.addRow(self.box_label, self.box_slider)
            self.speed_form.addRow(self.size_label, self.size_slider)
        if self.window.draw_segmentation:
            self.speed_form.addRow(self.opacity_label, self.opacity_slider)
            # TODO: separate boxes and points

        self.assessment_buttons = QHBoxLayout()
        self.good_button = QPushButton("Good")
        self.bad_button = QPushButton("Bad")
        self.good_button.clicked.connect(self.window.on_good)
        self.bad_button.clicked.connect(self.window.on_bad)
        self.assessment_buttons.addWidget(self.bad_button)
        self.assessment_buttons.addWidget(self.good_button)
        self.good_button.setVisible(False)
        self.bad_button.setVisible(False)

        self.video_buttons = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.prev_button = QPushButton("Previous")
        self.next_button.clicked.connect(self.window.on_next)
        self.prev_button.clicked.connect(self.window.on_prev)
        self.video_buttons.addWidget(self.prev_button)
        self.video_buttons.addWidget(self.next_button)
        if self.window.sequential or self.window.al_mode:
            self.next_button.setVisible(True)
            self.prev_button.setVisible(True)
        else:
            self.next_button.setVisible(False)
            self.prev_button.setVisible(False)

        self.back_button = QPushButton("Back to categories")
        self.back_button.clicked.connect(
            lambda: self.window.set_active_list("categories")
        )
        if self.window.active_list == "base":
            self.back_button.setVisible(False)
        elif self.window.active_list == "categories":
            self.back_button.setEnabled(False)

        if self.window.draw_segmentation:
            self.layout.addWidget(self.seglabel)
            self.layout.addWidget(self.seglist, 30)
        if len(self.window.animals) != 1:
            self.layout.addWidget(self.indlabel)
            self.layout.addWidget(self.animallist, 30)
        self.layout.addWidget(self.catlabel)
        self.layout.addWidget(self.catlist, 70)
        self.layout.addWidget(self.back_button)
        self.layout.addLayout(self.speed_form)
        self.layout.addWidget(self.correct_button)
        self.layout.addLayout(self.video_buttons)
        self.layout.addLayout(self.assessment_buttons)
        self.setLayout(self.layout)

    def set_buttons(self, ass, method):
        if ass and method == "good/bad":
            self.prev_button.setVisible(False)
            self.next_button.setVisible(False)
            self.good_button.setVisible(True)
            self.bad_button.setVisible(True)
        else:
            self.prev_button.setVisible(True)
            self.next_button.setVisible(True)
            self.good_button.setVisible(False)
            self.bad_button.setVisible(False)
            self.prev_button.disconnect()
            self.next_button.disconnect()
            if ass and method == "edit %":
                self.prev_button.clicked.connect(self.window.on_edit_prev)
                self.next_button.clicked.connect(self.window.on_edit_next)
            else:
                self.prev_button.clicked.connect(self.window.on_prev)
                self.next_button.clicked.connect(self.window.on_next)
