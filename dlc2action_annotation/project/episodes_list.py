import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QVBoxLayout,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
)
from dlc2action_annotation.widgets.viewer import Viewer as Viewer
from dlc2action.project import Project
from PyQt5.QtCore import Qt
from dlc2action_annotation.project.project_settings import ProjectSettings


class EpisodesList(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = project
        scroll = QScrollArea()
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        self.table = self.set_table()
        scroll.setWidget(self.table)
    
    def set_table(self):
        df = self.project.list_episodes()
        metrics = [x[1] for x in df.columns if x[0] == "results"]
        episodes = df.index
        table = QTableWidget(len(episodes), len(metrics) + 1)
        table.setHorizontalHeaderLabels(["Name"] + metrics)
        for i, episode in enumerate(episodes):
            table.setItem(i, 0, QTableWidgetItem(episode))
            for j, metric in enumerate(metrics):
                table.setItem(i, j + 1, QTableWidgetItem(str(df.loc[episode, ("results", metric)])))    
                table.item(i, j).setFlags(Qt.ItemIsEnabled)
        table.itemClicked.connect(self.onClicked)
        return table
    
    def onClicked(self, item):
        if item.column() != 0:
            return
        episode = item.text()
        settings = self.project._read_parameters()
        settings_episode = self.project._episodes().load_parameters(episode)
        settings = self.project._update(settings, settings_episode)
        self.window = ProjectSettings(settings, enabled=False, title=episode)
        self.window.show()


def main():
    app = QApplication(sys.argv)

    project = Project("oft")
    window = EpisodesList(project)
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()