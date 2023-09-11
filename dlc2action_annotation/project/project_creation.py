import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QFileDialog
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
import os
from dlc2action_annotation.widgets.viewer import Viewer as Viewer
from PyQt5.QtGui import QPixmap
from pathlib import Path


class ProjectCreation(QWidget):
    def __init__(self):
        super().__init__()
        self.projects_path = os.path.join(str(Path.home()), "DLC2Action")
        self.project_names = []
        if os.path.exists(self.projects_path):
            self.project_names = [x for x in os.listdir(self.projects_path) if os.path.isdir(os.path.join(self.projects_path, x))]
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        self.combo = self.make_combo()
        button = self.make_button()
        pic = self.make_logo()
        path_finder = self.make_path_choice()
        self.layout.addWidget(pic)
        self.layout.addSpacing(50)
        self.layout.addLayout(path_finder)
        self.layout.addWidget(self.combo)
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
        label_widget.setText(os.path.basename(file))
        self.projects_path = file
        label_widget.setText(self.projects_path)
        self.change_path()
    
    def make_combo(self):
        combo = QComboBox()
        for o in self.project_names:
            combo.addItem(o)
        combo.addItem("New Project")
        return combo
    
    def change_path(self):
        self.project_names = []
        if os.path.exists(self.projects_path):
            self.project_names = [x for x in os.listdir(self.projects_path) if os.path.isdir(os.path.join(self.projects_path, x))]
        self.combo.clear()
        for o in self.project_names:
            self.combo.addItem(o)
        self.combo.addItem("New Project")
    
    def make_button(self):
        button = QPushButton("Start")
        button.clicked.connect(self.create_project)
        return button
    
    def make_logo(self):
        pic = QLabel()
        pixmap = QPixmap("icons/horizontal_logo.png")
        
        # Calculate the width to fit the QLabel's size while keeping proportions
        desired_width = pic.width()
        scaled_pixmap = pixmap.scaledToWidth(desired_width, mode=Qt.SmoothTransformation)
        
        pic.setPixmap(scaled_pixmap)
        pic.setScaledContents(True)
        pic.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        return pic

    def create_project(self):
        print("Creating project")

    def sizeHint(self):
        return QSize(400, 300)


def main():
    app = QApplication(sys.argv)

    window = ProjectCreation()
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()