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
    QFormLayout,
    QCheckBox,
    QDialogButtonBox,
    QMenu,
    QLabel,
)
from dlc2action_annotation.widgets.viewer import Viewer as Viewer
from dlc2action.project import Project
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QEvent
from dlc2action_annotation.project.project_settings import ProjectSettings
from dlc2action_annotation.project.episode_training import EpisodeTraining
from dlc2action_annotation.project.utils import show_error, show_warning
from dlc2action_annotation.annotator import MainWindow as Annotator
import os
import mimetypes


class VideoChoice(QWidget):
    accepted = pyqtSignal(list)

    def __init__(self, options):
        super().__init__()
        self.options = options
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout = QFormLayout()
        scroll = QScrollArea(self)
        layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)

        scrollContent.setLayout(self.layout)
        scroll.setWidget(scrollContent)

        self.checkboxes = []
        for option in options:
            checkbox = QCheckBox()
            self.layout.addRow(option, checkbox)
            self.checkboxes.append(checkbox)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

    def accept(self):
        self.accepted.emit([self.options[i] for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()])
        self.close()
    
    def reject(self):
        self.close()


class EpisodesList(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = project
        annotation_type = self.project._read_parameters()["general"]["annotation_type"]
        self.can_annotate = (annotation_type == "dlc")
        self.setWindowTitle("Project: " + self.project.name)
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
        self.new_annotation_button = QPushButton("Annotate more videos")
        self.annotation_button.setEnabled(False)
        self.annotation_button.clicked.connect(self.generate_annotations)
        self.new_annotation_button.clicked.connect(self.annotate_more_videos)
        name_layout.addWidget(self.annotation_button)
        name_layout.addWidget(self.new_annotation_button)
        name_layout.addStretch(1)
        self.episode_le = QLineEdit()
        name_layout.addWidget(self.episode_le)
        self.create_episode_button = QPushButton("Create episode")
        name_layout.addWidget(self.create_episode_button)
        self.create_episode_button.clicked.connect(self.make_new_episode)
        self.update_settings_button = QPushButton("Update settings")
        self.update_settings_button.clicked.connect(self.update_settings)
        name_layout.addStretch(1)
        name_layout.addWidget(self.update_settings_button)
        return name_widget
    
    def make_new_episode(self):
        name = self.episode_le.text()
        if name == "":
            show_error("Please enter a name")
        title = f"Project: {self.project.name}, Episode: {name}"
        self.settings_window = ProjectSettings(self.project._read_parameters(), enabled=True, title=title, project=self.project)
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
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self.show_menu)
        table.selectionModel().selectionChanged.connect(self.row_selected)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.viewport().installEventFilter(self)
        return table
    
    def row_selected(self):
        self.annotation_button.setEnabled(True)

    def show_menu(self, item=None):
        self.menu.exec_(self.table.mapToGlobal(item))

    def show_parameters(self):
        episode = self.clicked_episode
        settings = self.project._read_parameters()
        settings_episode = self.project._episodes().load_parameters(episode)
        settings = self.project._update(settings, settings_episode)
        self.window = ProjectSettings(settings, enabled=False, title=episode)
        self.window.show()

    def _get_eligible_videos(self):
        settings = self.project._read_parameters()
        data_path = settings["data"]["data_path"]
        if isinstance(data_path, str):
            data_path = [data_path]
        potential_files = []
        for path in data_path:
            potential_files.extend([os.path.join(path, file) for file in os.listdir(path)])
        data_suffix = settings["data"]["data_suffix"]
        if isinstance(data_suffix, str):
            data_suffix = [data_suffix]
        videos = []
        for file in potential_files:
            guess_type = mimetypes.guess_type(file)[0]
            if guess_type is None or not guess_type.startswith('video'):
                continue
            name = file.split(".")[0]
            potential_data_files = [name + suffix for suffix in data_suffix]
            if any([os.path.exists(potential_data_file) for potential_data_file in potential_data_files]):
                videos.append(file)
        return videos

    def generate_annotations(self):
        if not self.can_annotate:
            show_warning(
                "Different annotation types",
                "Since the annotation type of this project is not DLC, the annotation output cannot be used directly for training."
            )
        row = self.table.selectionModel().selectedRows()[0].row()
        episode = self.table.item(row, 0).text()
        videos = self._get_eligible_videos()
        if len(videos) == 0:
            show_error("No videos found")
            return
        self.video_choice = VideoChoice(videos)
        self.video_choice.accepted.connect(lambda x: self.annotate_with_suggestion(episode, x))
        self.video_choice.show()

    def annotate_with_suggestion(self, episode, videos):
        settings = self.project._read_parameters()
        data_suffix = settings["data"]["data_suffix"]
        if isinstance(data_suffix, str):
            data_suffix = [data_suffix[0]]
        data_files = []
        for video in videos:
            name = video.split('.')[0]
            for suffix in data_suffix:
                if os.path.exists(name + suffix):
                    data_files.append(name + suffix)
                    break
        self.project.remove_saved_features()
        self.project.run_suggestion(
            episode, 
            suggestion_episodes=[episode], 
            suggestion_classes=["Grooming", "Supported", "Unsupported"], 
            force=True, 
            file_paths=data_files,
            parameters_update={"general": {"only_load_annotated": False}}
        )
        self.annotate_videos(videos, suggestion=episode)

    def update_settings(self):
        self.settings = ProjectSettings(self.project._read_parameters())
        self.settings.show()
        self.settings.accepted.connect(lambda x: self.project.update_parameters(x))
    
    def annotate_more_videos(self):
        if not self.can_annotate:
            show_warning(
                "Different annotation types"
                "Since the annotation type of this project is not DLC, the annotation output cannot be used directly for training."
            )
        videos = self._get_eligible_videos()
        if len(videos) == 0:
            show_error("No videos found")
            return
        self.video_choice = VideoChoice(videos)
        self.video_choice.accepted.connect(self.annotate_videos)
        self.video_choice.show()

    def annotate_videos(self, videos, suggestion=None):
        if suggestion is not None:
            suggestion_files = [self.project._suggestion_path(os.path.basename(name).split('.')[0], suggestion) for name in videos]
        else:
            suggestion_files = None
        annotation_path = self.project._read_parameters()["data"]["annotation_path"]
        annotation_suffix = self.project._read_parameters()["data"]["annotation_suffix"]
        if not isinstance(annotation_path, str):
            annotation_path = annotation_path[0]
        if not isinstance(annotation_suffix, str):
            annotation_suffix = annotation_suffix[0]
        annotation_files = [os.path.join(annotation_path, os.path.basename(name).split('.')[0] + annotation_suffix) for name in videos]
        window = Annotator(
            videos=videos,
            output_file=None,
            multiview=False,
            dev=False,
            active_learning=False,
            config_file="config.yaml",
            suggestion_files=suggestion_files,
            annotation_files=annotation_files,
        )
        window.show()
        self.close()

    def sizeHint(self):
        return QSize(900, 700)
    
    def eventFilter(self, source, event):
        if(event.type() == QEvent.MouseButtonPress and
            event.buttons() == Qt.RightButton and
            source is self.table.viewport()):
            item = self.table.itemAt(event.pos())
            if item is not None:
                self.menu = QMenu(self)
                self.show_action = self.menu.addAction("Show parameters")         #(QAction('test'))
                self.show_action.triggered.connect(self.show_parameters)
                row = item.row()
                self.clicked_episode = self.table.item(row, 0).text()
                #menu.exec_(event.globalPos())
        return super().eventFilter(source, event)


def main():
    app = QApplication(sys.argv)

    project = Project("oft")
    window = EpisodesList(project)
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()