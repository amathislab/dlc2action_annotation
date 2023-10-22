import os
import sys
from pathlib import Path

from dlc2action.project import Project
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dlc2action_annotation.project.project_display import EpisodesList
from dlc2action_annotation.project.project_settings import ProjectSettings, TypeChoice
from dlc2action_annotation.project.utils import show_error
from dlc2action_annotation.utils import get_library_path
from dlc2action_annotation.widgets.viewer import Viewer as Viewer


class StartingScreen(QWidget):
    """Widget for choosing / creating a project."""

    def __init__(self):
        super().__init__()
        self.projects_path = os.path.join(str(Path.home()), "DLC2Action")
        self.project_names = []
        if os.path.exists(self.projects_path):
            self.project_names = [
                x
                for x in os.listdir(self.projects_path)
                if os.path.isdir(os.path.join(self.projects_path, x))
            ]
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        self.combo = self.make_combo()
        button = self.make_button()
        pic = self.make_logo()
        # path_finder = self.make_path_choice()
        self.name = self.make_name()
        self.data_path_finder = self.make_data_path_finder()
        self.annotation_path_finder = self.make_annotation_path_finder()
        self.layout.addWidget(pic)
        self.layout.addSpacing(50)
        # self.layout.addLayout(path_finder)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.name)
        self.layout.addWidget(self.data_path_finder)
        self.layout.addWidget(self.annotation_path_finder)
        self.layout.addWidget(button)
        self.setLayout(self.layout)

    def make_path_choice(self):
        layout = QHBoxLayout()
        button = QPushButton("Browse")
        button.setMaximumWidth(100)
        label = QLabel(self.projects_path)
        button.clicked.connect(lambda: self.get_dir(label))
        layout.addWidget(label)
        layout.addWidget(button)
        return layout

    def get_dir(self, label_widget):
        file = QFileDialog().getExistingDirectory(self)
        label_widget.setText(file)

    def make_combo(self):
        combo = QComboBox()
        for o in self.project_names:
            combo.addItem(o)
        combo.addItem("New Project")
        combo.currentTextChanged.connect(self.activate_name)
        return combo

    def make_name(self):
        name = QWidget()
        label = QLabel("Project Name:")
        self.name_le = QLineEdit()
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.name_le)
        name.setLayout(layout)
        if not self.combo.currentText() == "New Project":
            name.setVisible(False)
        return name

    def make_data_path_finder(self):
        widget = QWidget()
        layout = QHBoxLayout()
        label = QLabel("Data Path:")
        button = QPushButton("Browse")
        button.setMaximumWidth(100)
        self.data_label = QLabel("None")
        button.clicked.connect(lambda: self.get_dir(self.data_label))
        layout.addWidget(label)
        layout.addWidget(self.data_label)
        layout.addWidget(button)
        widget.setLayout(layout)
        if not self.combo.currentText() == "New Project":
            widget.setVisible(False)
        return widget

    def make_annotation_path_finder(self):
        widget = QWidget()
        layout = QHBoxLayout()
        label = QLabel("Annotation Path:")
        button = QPushButton("Browse")
        button.setMaximumWidth(100)
        self.annotation_label = QLabel("None")
        button.clicked.connect(lambda: self.get_dir(self.annotation_label))
        layout.addWidget(label)
        layout.addWidget(self.annotation_label)
        layout.addWidget(button)
        widget.setLayout(layout)
        if not self.combo.currentText() == "New Project":
            widget.setVisible(False)
        return widget

    def activate_name(self, text):
        if text == "New Project":
            self.name.setVisible(True)
            self.data_path_finder.setVisible(True)
            self.annotation_path_finder.setVisible(True)
        else:
            self.name.setVisible(False)
            self.data_path_finder.setVisible(False)
            self.annotation_path_finder.setVisible(False)

    def change_path(self):
        self.project_names = []
        if os.path.exists(self.projects_path):
            self.project_names = [
                x
                for x in os.listdir(self.projects_path)
                if os.path.isdir(os.path.join(self.projects_path, x))
            ]
        self.combo.clear()
        for o in self.project_names:
            self.combo.addItem(o)
        self.combo.addItem("New Project")

    def make_button(self):
        button = QPushButton("Start")
        button.clicked.connect(self.start_project)
        return button

    def make_logo(self):
        pic = QLabel()
        icon_path = os.path.join(get_library_path(), "img")
        logo_path = os.path.join(icon_path, "horizontal_logo.png")
        pixmap = QPixmap(logo_path)

        # Calculate the width to fit the QLabel's size while keeping proportions
        desired_width = pic.width()
        scaled_pixmap = pixmap.scaledToWidth(
            desired_width, mode=Qt.SmoothTransformation
        )

        pic.setPixmap(scaled_pixmap)
        pic.setScaledContents(True)
        pic.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        return pic

    def make_new_project(self):
        name = self.name_le.text()
        if name == "":
            show_error("Please enter a name for the project")
            return
        if name in os.listdir(self.projects_path):
            show_error("Project already exists")
            return
        if not os.path.exists(self.data_label.text()):
            show_error("Please select a data path")
            return
        if not os.path.exists(self.annotation_label.text()):
            show_error("Please select an annotation path")
            return
        self.type_choice = TypeChoice()
        self.type_choice.show()
        self.type_choice.accepted.connect(
            lambda x: self.set_parameters(name, x[0], x[1])
        )
        self.close()

    def set_parameters(self, name, data_type, annotation_type):
        project = Project(
            name,
            projects_path=self.projects_path,
            data_type=data_type,
            annotation_type=annotation_type,
            data_path=self.data_label.text(),
            annotation_path=self.annotation_label.text(),
        )
        self.settings = ProjectSettings(project._read_parameters(catch_blanks=False))
        self.settings.show()
        self.settings.accepted.connect(lambda x: self.start_new_project(project, x))

    def start_new_project(self, project, settings):
        project.update_parameters(settings)
        self.window = EpisodesList(project)
        self.window.show()
        self.close()

    def start_project(self):
        choice = self.combo.currentText()
        if choice == "New Project":
            self.make_new_project()
        else:
            project = Project(choice, projects_path=self.projects_path)
            self.window = EpisodesList(project)
            self.window.show()
            self.close()

    def sizeHint(self):
        return QSize(400, 300)


def main():
    app = QApplication(sys.argv)

    window = StartingScreen()
    window.show()

    app.exec_()


if __name__ == "__main__":
    main()
