#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
from PyQt5.QtWidgets import (
    QDialog,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QCheckBox,
    QScrollArea,
    QComboBox,
    QSlider,
    QRadioButton,
    QDialogButtonBox,
)
from PyQt5.QtGui import QPixmap, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal


class LineEdit(QLineEdit):
    next_field = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, line, window, *args, **kwargs):
        super(LineEdit, self).__init__(*args, **kwargs)
        self.line = line
        self.window = window

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            shift = True
        else:
            shift = False
        if event.key() == Qt.Key_Enter or event.key() == 16777220:
            if shift:
                self.finished.emit()
            else:
                self.next_field.emit()
        else:
            super(LineEdit, self).keyPressEvent(event)


class CatLine(QWidget):
    next_line = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, window, col, name, sc, lines, hot_buttons):
        super(CatLine, self).__init__()
        self.window = window
        self.layout = QHBoxLayout()
        self.button = QLabel()
        pixmap = QPixmap(100, 100)
        pixmap.fill(col)
        self.button.setPixmap(pixmap)
        self.name_field = LineEdit(self, self.window, self)
        self.name_field.setText(name)
        self.name_field.textChanged.connect(self.on_text)
        self.name_field.next_field.connect(self.switch)
        self.name_field.finished.connect(self.end)
        self.sc_field = LineEdit(self, self.window, self)
        self.sc_field.setText(sc)
        self.sc_field.textChanged.connect(self.set_sc)
        self.sc_field.finished.connect(self.end)
        fm = self.sc_field.fontMetrics()
        m = self.sc_field.textMargins()
        c = self.sc_field.contentsMargins()
        w = 2 * fm.width("X") + m.left() + m.right() + c.left() + c.right() + 2
        self.sc_field.setMaximumWidth(w)
        self.sc_field.next_field.connect(self.next)
        self.button.setFixedSize(w, w)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.name_field)
        self.layout.addWidget(self.sc_field)
        self.setLayout(self.layout)
        self.n = len(lines)
        self.lines = lines
        self.hot_buttons = hot_buttons

    def on_text(self, event):
        text = self.name_field.text()
        if (
            len(self.lines) > 0
            and text != ""
            and self.lines[self.n - 1].name_field.text() == ""
            and self.n != 0
        ):
            self.window.show_warning("line")
            self.lines[self.n - 1].name_field.setFocus()
            self.name_field.setText("")
        elif len(text) > 0:
            sc = self.name_field.text()[0].upper()
            hot_buttons = [
                x.sc_field.text()
                for x in self.lines
                if x.sc_field.text() != "" and x != self
            ] + self.hot_buttons
            if sc not in hot_buttons:
                self.sc_field.setText(sc)

    def set_sc(self, value):
        hot_buttons = [
            x.sc_field.text()
            for x in self.lines
            if x.sc_field.text() != "" and x != self
        ] + self.hot_buttons
        value = value.upper()
        if len(value) > 1:
            self.sc_field.setText(self.sc_field.text()[:-1])
            self.window.show_warning("long")
        elif value in hot_buttons:
            self.sc_field.setText("")
            self.window.show_warning("used", value)
        else:
            self.sc_field.setText(value)

    def switch(self):
        if len(self.sc_field.text()) > 0:
            self.next_line.emit(self.n)
        else:
            self.sc_field.setFocus()

    def next(self):
        self.next_line.emit(self.n)

    def end(self):
        self.finished.emit()


class CatDialog(QDialog):
    def __init__(self, catDict, shortCut, invisible, key, *args, **kwargs):
        super(CatDialog, self).__init__(*args, **kwargs)
        self.shortCut = shortCut
        self.catDict = catDict
        self.key = key
        self.cat_list = {}
        self.invisible = invisible
        self.hot_buttons = []
        self.colors = [
            QColor(51, 204, 51),
            QColor(20, 170, 255),
            QColor(153, 102, 255),
            QColor(255, 102, 0),
            QColor(102, 255, 178),
            QColor(100, 0, 170),
            QColor(20, 20, 140),
            QColor(174, 50, 160),
            QColor(180, 0, 0),
            QColor(255, 255, 0),
            QColor(128, 195, 66),
            QColor(65, 204, 51),
            QColor(20, 180, 255),
            QColor(153, 102, 230),
            QColor(240, 102, 0),
            QColor(120, 255, 178),
            QColor(100, 10, 170),
            QColor(20, 30, 140),
            QColor(150, 50, 160),
            QColor(180, 0, 20),
            QColor(220, 255, 0),
            QColor(150, 195, 66),
        ]

        self.layout = QVBoxLayout()
        self.label = QVBoxLayout()
        if self.key != "base":
            self.key_label = QLabel(self.key)
            myfont = QFont()
            myfont.setBold(True)
            self.key_label.setFont(myfont)
            self.label.addWidget(self.key_label)
        self.text = QLabel(
            "Here you can edit the label names and the shortcuts. \n"
            "Press Enter to add a new line, Shift+Enter to move on."
        )
        self.label.addWidget(self.text)
        self.layout.addLayout(self.label)

        self.line_layout = QVBoxLayout()
        self.line_widget = QWidget()
        self.line_widget.setLayout(self.line_layout)
        self.line_scroll = QScrollArea()
        self.line_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.line_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.line_scroll.setWidgetResizable(True)
        self.line_scroll.setWidget(self.line_widget)
        scroll_bar = self.line_scroll.verticalScrollBar()
        scroll_bar.rangeChanged.connect(
            lambda: scroll_bar.setValue(scroll_bar.maximum())
        )
        self.create_lines(main_key=self.key)
        self.max_key = max(list(self.catDict["base"].keys()))
        self.layout.addWidget(self.line_scroll)

        self.new_button = QPushButton("Add label")
        self.new_button.clicked.connect(self.add_line)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.finish)
        self.layout.addWidget(self.new_button)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def show_warning(self, t, value=None):
        msg = QMessageBox()
        if t == "used":
            msg.setText(f"The {value} shortcut is already in use!")
        elif t == "long":
            msg.setText("Shortcuts need to be one letter long!")
        elif t == "line":
            msg.setText("You need to fill all previous lines first!")
        elif t == "category":
            msg.setText(f"You already have a {value} label in a different category!")
        msg.setWindowTitle("Warning")
        msg.addButton(QMessageBox.Ok)
        msg.exec_()

    def add_line(self, ind=None, name="", sc=""):
        if type(ind) is not int:
            ind = self.max_key + 1
            self.max_key += 1
        if type(name) is not str:
            name = ""
            sc = ""
        col = self.colors[ind % len(self.colors)]
        line = CatLine(self, col, name, sc, self.lines, self.hot_buttons)
        line.next_line.connect(self.new_line)
        line.finished.connect(self.finish)
        self.lines.append(line)
        self.line_layout.addWidget(line)
        line.name_field.setFocus()
        self.update()
        # self.cat_list[ind] = line

    def new_line(self, n):
        if n + 1 < len(self.lines):
            self.lines[n + 1].name_field.setFocus()
        else:
            self.add_line()
            self.lines[-1].name_field.setFocus()

    def create_lines(self, main_key="base"):
        self.lines = []
        for cat in self.catDict[main_key]:
            if self.catDict[main_key][cat] not in self.invisible:
                sc = ""
                for key in self.shortCut[main_key]:
                    if self.shortCut[main_key][key] == cat:
                        sc = key
                self.add_line(cat, self.catDict[main_key][cat], sc)
                self.cat_list[cat] = self.lines[-1]
        if len(self.catDict[main_key]) == 0:
            self.add_line()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            shift = True
        else:
            shift = False
        if event.key() == Qt.Key_Enter or event.key() == 16777220:
            if shift:
                self.finish()
        else:
            super(CatDialog, self).keyPressEvent(event)

    def add_action(self, text, i, sc):
        self.catDict[self.key][i] = text
        self.catDict["base"][i] = text
        self.actions.append(text)
        if self.key == "categories" and text not in self.catDict.keys():
            self.catDict[text] = {}
        if len(sc) == 1:
            self.shortCut[self.key][sc] = i

    def finish(self, event=None):
        main_key = self.key
        self.actions = []
        self.shortCut[main_key] = {}
        taken = [""] + self.invisible
        for i, line in self.cat_list.items():
            text = line.name_field.text()
            if text in self.invisible:
                self.invisible.remove(text)
            cat = self.catDict[main_key][i]
            sc = line.sc_field.text()
            if text == "":
                self.invisible.append(cat)
            else:
                self.add_action(text, i, sc)
                taken.append(text)
        max_key = max(list(self.catDict["base"].keys()))
        for line in self.lines:
            text = line.name_field.text()
            if text in self.invisible:
                self.actions.append(text)
                self.invisible.remove(text)
            sc = line.sc_field.text()
            if text not in taken:
                max_key += 1
                self.add_action(text, max_key, sc)
        self.accept()

    def exec_(self):
        super(CatDialog, self).exec_()
        return (self.catDict, self.shortCut, self.invisible, self.actions)


class LoadDialog(QDialog):
    def __init__(self, filename):
        super(LoadDialog, self).__init__()
        self.boxes = None
        self.skeleton = None
        self.labels = None

        self.label = QLabel(
            f"Would you like to load some additional files for {filename}?"
        )
        self.boxes_button = QPushButton("Load bounding boxes")
        self.skeleton_button = QPushButton("Load DLC output")
        self.labels_button = QPushButton("Load annotation")
        self.no_button = QPushButton("Continue")

        self.boxes_button.clicked.connect(self.load_boxes)
        self.skeleton_button.clicked.connect(self.load_skeleton)
        self.labels_button.clicked.connect(self.load_annotation)
        self.no_button.clicked.connect(self.accept)

        self.no_button.setDefault(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.boxes_button)
        self.layout.addWidget(self.skeleton_button)
        self.layout.addWidget(self.labels_button)
        self.layout.addWidget(self.no_button)
        self.setLayout(self.layout)

    def load_boxes(self):
        self.boxes = [QFileDialog.getOpenFileName(self, "Open file")[0]]
        if len(self.boxes[0]) == 0:
            self.boxes = None
        else:
            self.boxes_button.setEnabled(False)
            self.skeleton_button.setEnabled(False)

    def load_skeleton(self):
        self.skeleton = [QFileDialog.getOpenFileName(self, "Open file")[0]]
        if len(self.skeleton[0]) == 0:
            self.skeleton = None
        else:
            self.boxes_button.setEnabled(False)
            self.skeleton_button.setEnabled(False)

    def load_annotation(self):
        self.labels = QFileDialog.getOpenFileName(
            self, "Open file", filter=("Annotation files (*.h5 *.pickle)")
        )[0]
        if len(self.labels) == 0:
            self.labels = None
        self.labels_button.setEnabled(False)

    def exec_(self):
        super(LoadDialog, self).exec_()
        return (self.boxes, self.skeleton, self.labels)


class ChoiceDialog(QDialog):
    def __init__(self, action_dict, filename):
        super(ChoiceDialog, self).__init__()
        self.actions = None
        self.display_cats = None
        self.action_dict = action_dict
        self.button = QPushButton("OK")
        self.button.clicked.connect(self.finish)
        self.cats_checkbox = QCheckBox("Nested annotation")
        self.cats_checkbox.setChecked(False)
        self.label = QLabel(f"Choose the actions you want to annotate in {filename}")
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self.on_change)
        for category in action_dict:
            parent = QTreeWidgetItem(self.tree)
            parent.setText(0, category)
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            for action in action_dict[category]:
                child = QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setText(0, action)
                child.setCheckState(0, Qt.Unchecked)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.tree)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.cats_checkbox)
        self.setLayout(self.layout)

    def finish(self, event):
        self.actions = []
        titles = []
        iterator = QTreeWidgetItemIterator(self.tree, QTreeWidgetItemIterator.Checked)
        while iterator.value():
            item = iterator.value()
            if item.text(0) not in self.action_dict.keys():
                self.actions[-1].append(item.text(0))
            else:
                self.actions.append([])
                titles.append(item.text(0))
            iterator += 1
        if len(titles) > 0:
            self.actions = {k: v for k, v in zip(titles, self.actions)}
        self.display_cats = self.cats_checkbox.isChecked()
        self.accept()

    def on_change(self, event):
        iterator = QTreeWidgetItemIterator(self.tree, QTreeWidgetItemIterator.Checked)
        count = 0
        while iterator.value():
            item = iterator.value()
            if item.text(0) in self.action_dict.keys():
                count += 1
            iterator += 1

        self.cats_checkbox.setChecked(count > 1)

    def exec_(self):
        super(ChoiceDialog, self).exec_()
        return self.actions, self.display_cats


class ChoiceDialogExample(ChoiceDialog):
    def __init__(self, action_dict):
        super(ChoiceDialogExample, self).__init__(action_dict, "")
        self.label.setText(f"Choose the actions you want to sample from")
        self.cats_checkbox.setVisible(False)

    def exec_(self):
        super(ChoiceDialog, self).exec_()
        if isinstance(self.actions, dict):
            actions = []
            for a in self.actions.values():
                actions += a
            self.actions = actions
        return self.actions


class AssessmentDialog(QDialog):
    def __init__(self, sampler):
        super(AssessmentDialog, self).__init__()
        self.sampler = sampler
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.combo = QComboBox()
        self.combo.addItem("good/bad")
        self.combo.addItem("edit %")
        self.combo.setCurrentIndex(["good/bad", "edit %"].index(sampler.get_method()))
        self.combo.currentTextChanged.connect(self.method_changed)
        self.threshold_value = QLabel(f"{sampler.get_threshold():.2f}")
        self.threshold_field = QSlider(Qt.Horizontal)
        self.threshold_field.setMinimum(0)
        self.threshold_field.setMaximum(100)
        self.threshold_field.setSingleStep(5)
        self.threshold_field.setTickInterval(50)
        self.threshold_field.setValue(sampler.get_threshold() * 100)
        self.threshold_field.valueChanged.connect(self.threshold_changed)
        self.combo_layout = QVBoxLayout()
        self.combo_layout.addWidget(QLabel("Method:"))
        self.combo_layout.addWidget(self.combo)
        self.threshold_widget = QWidget()
        self.threshold_layout = QVBoxLayout()
        self.threshold_widget.setLayout(self.threshold_layout)
        threshold_labels = QHBoxLayout()
        threshold_labels.addWidget(QLabel("Threshold value:"))
        threshold_labels.addWidget(self.threshold_value)
        self.threshold_layout.addLayout(threshold_labels)
        self.threshold_layout.addWidget(self.threshold_field)
        self.threshold_widget.setVisible(sampler.get_method() == "edit %")
        self.controls_layout = QHBoxLayout()
        self.controls_layout.addLayout(self.combo_layout)
        self.controls_layout.addWidget(self.threshold_widget)
        self.line_layout = QFormLayout()
        self.line_widget = QWidget()
        self.line_widget.setLayout(self.line_layout)
        self.line_scroll = QScrollArea()
        self.line_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.line_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.line_scroll.setWidgetResizable(True)
        self.line_scroll.setWidget(self.line_widget)
        self.layout.addLayout(self.controls_layout)
        self.layout.addWidget(self.line_scroll)
        self.chosen_label = None
        self.print_values()

    def print_values(self):
        self.clearLayout(self.line_layout)
        values = self.sampler.compute()
        labels = sorted(values.keys())
        for label in labels:
            if label in values:
                value = values[label]
            else:
                value = None
            self.add_row(label, value)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def method_changed(self, method):
        self.sampler.set_method(method)
        self.print_values()
        self.threshold_widget.setVisible(self.sampler.get_method() == "edit %")

    def threshold_changed(self, value):
        value /= 100
        self.sampler.set_threshold(value)
        self.threshold_value.setText(f"{value:.2f}")
        self.print_values()

    def add_row(self, label, value):
        l = QHBoxLayout()
        l.addWidget(QLabel(label))
        if value is None:
            value = "?/?"
        l.addWidget(QLabel(value))
        button = QPushButton("Sample")
        l.addWidget(button)
        button.clicked.connect(lambda x: self.emit_label(label))
        l.addWidget(button)
        self.line_layout.addRow(l)

    def emit_label(self, label):
        self.chosen_label = label
        self.accept()

    def exec_(self):
        super().exec_()
        return self.chosen_label


class Form(QDialog):
    def __init__(self, videos, parent=None):
        super(Form, self).__init__(parent)
        # Create widgets
        self.label = QLabel("Which video does this skeleton file relate to?")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.buttons = [QRadioButton(video) for video in videos]
        for button in self.buttons:
            layout.addWidget(button)
        # Set dialog layout
        self.setLayout(layout)
        self.videos = videos
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)

    def exec_(self):
        super().exec_()
        for i, button in enumerate(self.buttons):
            if button.isChecked():
                return self.videos[i]
