import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QAbstractItemView,
)
from dlc2action_annotation.widgets.viewer import Viewer as Viewer
from dlc2action.project import Project
from PyQt5.QtCore import Qt, QSize
from dlc2action_annotation.project.project_settings import ProjectSettings
from dlc2action_annotation.project.episode_training import EpisodeTraining
from dlc2action_annotation.project.utils import show_error
from dlc2action_annotation.annotator import MainWindow as Annotator
import os
import mimetypes


class EpisodesList(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = project
        self.scroll_area = QScrollArea()
        self.layout = QVBoxLayout(self)
        self.name_field = self.set_name_field()
        self.layout.addWidget(self.scroll_area)
        self.scroll_area.setWidgetResizable(True)
        self.table = self.set_table()
        self.scroll_area.setWidget(self.table)
        self.layout.addWidget(self.name_field)

    def set_buttons(self):
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        new_episode_button = QPushButton("New episode")
        new_search_button = QPushButton("New search")
        buttons.addWidget(new_episode_button)
        buttons.addWidget(new_search_button)
        new_episode_button.clicked.connect(self.make_new_episode)
        new_search_button.clicked.connect(self.make_new_search)
        return buttons
    
    def set_name_field(self):
        name_widget = QWidget()
        name_layout = QHBoxLayout()
        name_widget.setLayout(name_layout)
        self.annotation_button = QPushButton("Generate annotations")
        self.annotation_button.setEnabled(False)
        self.annotation_button.clicked.connect(self.generate_annotations)
        name_layout.addWidget(self.annotation_button)
        name_layout.addStretch(1)
        self.episode_le = QLineEdit()
        name_layout.addWidget(self.episode_le)
        self.create_episode_button = QPushButton("Create episode")
        name_layout.addWidget(self.create_episode_button)
        self.create_episode_button.clicked.connect(self.make_new_episode)
        return name_widget
    
    def make_new_episode(self):
        name = self.episode_le.text()
        if name == "":
            show_error("Please enter a name")
        self.settings_window = ProjectSettings(self.project._read_parameters(), enabled=True, title=name, project=self.project)
        self.settings_window.show()
        self.settings_window.accepted.connect(lambda x: self.create_episode(name, x))
    
    def create_episode(self, name, settings):
        model_name = settings["general"]["model_name"]
        if model_name in self.project.list_searches().index:
            load_search = model_name
        else:
            load_search = None
        self.training_window = EpisodeTraining(self.project, name, episode_settings=settings, load_search=load_search)
        self.training_window.finished.connect(self.update_table)
        self.training_window.show()

    def update_table(self):
        table = self.set_table()
        self.scroll_area.setWidget(table)
        self.table = table
        self.update()
    
    def set_table(self):
        df = self.project.list_episodes()
        metrics = [x[1] for x in df.columns if x[0] == "results"]
        episodes = df.index
        table = QTableWidget(len(episodes), len(metrics) + 1)
        table.setHorizontalHeaderLabels(["Name"] + metrics)
        for i, episode in enumerate(episodes):
            table.setItem(i, 0, QTableWidgetItem(episode))
            for j, metric in enumerate(metrics):
                table.setItem(i, j + 1, QTableWidgetItem(str(df.loc[episode, ("results", metric)].round(3))))    
                # table.item(i, j).setFlags(Qt.ItemIsEnabled)
        table.itemClicked.connect(self.name_clicked)
        table.selectionModel().selectionChanged.connect(self.row_selected)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return table
    
    def row_selected(self):
        self.annotation_button.setEnabled(True)

    def name_clicked(self, item):
        if item.column() != 0:
            return
        episode = item.text()
        settings = self.project._read_parameters()
        settings_episode = self.project._episodes().load_parameters(episode)
        settings = self.project._update(settings, settings_episode)
        self.window = ProjectSettings(settings, enabled=False, title=episode)
        self.window.show()

    def generate_annotations(self):
        row = self.table.selectionModel().selectedRows()[0].row()
        episode = self.table.item(row, 0).text()
        settings = self.project._read_parameters()
        data_path = settings["data"]["data_path"]
        data_suffix = settings["data"]["data_suffix"]
        videos = []
        names = []
        data_files = []
        for file in os.listdir(data_path):
            guess_type = mimetypes.guess_type(file)[0]
            if guess_type is None or not guess_type.startswith('video'):
                continue
            name = file.split(".")[0]
            if name + data_suffix in os.listdir(data_path):
                videos.append(os.path.join(data_path, file))
                names.append(name)
                data_files.append(os.path.join(data_path, name + data_suffix))
        self.project.remove_saved_features()
        self.project.run_suggestion(
            episode, 
            suggestion_episodes=[episode], 
            suggestion_classes=["Grooming", "Supported", "Unsupported"], 
            force=True, 
            file_paths=data_files,
            parameters_update={"general": {"only_load_annotated": False}}
        )
        window = Annotator(
            videos=videos,
            output_file=None,
            multiview=False,
            dev=False,
            active_learning=False,
            show_settings=False,
            config_file="config.yaml",
            suggestion_files=[self.project._suggestion_path(name, episode) for name in names],
        )
        window.show()
        self.close()

    def sizeHint(self):
        return QSize(700, 500)


def main():
    app = QApplication(sys.argv)

    project = Project("oft")
    window = EpisodesList(project)
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()