import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QTabWidget,
    QCheckBox,
    QLineEdit,
    QSlider,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from dlc2action_annotation.widgets.viewer import Viewer as Viewer
from dlc2action_annotation.widgets.settings import MultipleInputWidget, MultipleDoubleInputWidget, CategoryInputWidget
from dlc2action.project import Project
from dlc2action.options import input_stores, annotation_stores
from qtwidgets import Toggle
import os


class EpisodeCreation(QWidget):
    def __init__(self, project):
        self.project = project
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Episode name"))
        self.episode_name = QLineEdit()
        self.layout().addWidget(self.episode_name)
        self.layout().addWidget(QLabel("Load search results"))
        self.search_results = QComboBox()
        for search_result in self.project.list_searches().index:
            self.search_results.addItem(search_result)
        self.search_results.addItem("-")
        self.layout().addWidget(self.search_results)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout().addWidget(self.button_box)

    def reject(self):
        self.close()

    def accept(self):
        episode_name = self.episode_name.text()
        if episode_name == "":
            QMessageBox.warning(self, "Warning", "Please enter an episode name")
            return
        if episode_name in self.project.list_episodes().index:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("This name is already taken")
            msg.setInformativeText("Do you want to overwrite the episode?")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                
            retval = msg.exec_()
            if retval == QMessageBox.Ok:
                print("Accept")
            else:
                return
        load_search = self.search_results.currentText()
        if load_search == "-":
            load_search = None
        print(f"Accept {episode_name=} {load_search=}")


def main():
    app = QApplication(sys.argv)

    project = Project("test")
    window = EpisodeCreation(project=project)
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()