from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtGui import QFont
from dlc2action.project import Project
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from time import sleep
from collections import defaultdict
import numpy as np
from dlc2action_annotation.utils import get_library_path


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, project, episode, episode_settings=None, load_search=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.episode = episode
        self.episode_settings = episode_settings
        self.load_search = load_search

    def run(self):
        """Long-running task."""
        self.project.run_episode(
            self.episode,
            parameters_update=self.episode_settings,
            force=True,
            load_search=self.load_search,
        )
        self.finished.emit()


class EpisodeTraining(QWidget):
    finished = pyqtSignal()

    def __init__(self, project, episode, episode_settings=None, load_search=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Project: " + project.name)
        self.tabs = QTabWidget()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.tabs)

        if episode_settings is None:
            episode_settings = project._read_parameters()
        self.project = project
        self.episode = episode
        self.episode_settings = episode_settings
        self.load_search = load_search

        self.valGraph = pg.PlotWidget()
        self.tabs.addTab(self.valGraph, "Validation metrics")
        self.valGraph.setBackground('w')
        self.valGraph.setXRange(0, self.episode_settings["training"]["num_epochs"], padding=0)
        self.valGraph.setYRange(0, 1, padding=0)
        self.valGraph.setLabel('bottom', 'Epoch', color='black', size="15pt")
        self.valGraph.setTitle(f"episode {episode}", color='black', size="15pt")
        self.val_data_lines = {}
        legend = self.valGraph.addLegend()
        legend.setLabelTextSize("15pt")

        self.trainGraph = pg.PlotWidget()
        self.tabs.addTab(self.trainGraph, "Training metrics")
        self.trainGraph.setBackground('w')
        self.trainGraph.setXRange(0, self.episode_settings["training"]["num_epochs"], padding=0)
        self.trainGraph.setYRange(0, 1, padding=0)
        self.trainGraph.setLabel('bottom', 'Epoch', color='black', size="15pt")
        self.trainGraph.setTitle(f"episode {episode}", color='black', size="15pt")
        self.train_data_lines = {}
        legend = self.trainGraph.addLegend()
        legend.setLabelTextSize("15pt")

        self.lossGraph = pg.PlotWidget()
        self.tabs.addTab(self.lossGraph, "Loss")
        self.lossGraph.setBackground('w')
        self.lossGraph.setXRange(0, self.episode_settings["training"]["num_epochs"], padding=0)
        self.lossGraph.setLabel('bottom', 'Epoch', color='black', size="15pt")
        self.lossGraph.setTitle(f"episode {episode}", color='black', size="15pt")
        self.loss_data_lines = {}
        legend = self.lossGraph.addLegend()
        legend.setLabelTextSize("15pt")

        colors_path = os.path.join(get_library_path(), "colors.txt")
        with open(colors_path) as f:
            self.colors = [[int(x) for x in line.strip().split()] for line in f.readlines()]
        self.color_index = 0
    
        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        self.run_episode()

    def run_episode(self):
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.project, self.episode, self.episode_settings, self.load_search)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        self.thread.finished.connect(
            self.finish
        )

    def finish(self):
        print('THREAD FINISHED')
        self.finished.emit()
        self.close()

    def get_metric_log(self, mode: str):
        """Get the metric log.

        Parameters
        ----------
        mode : {'train', 'val'}
            the mode to get the log from
        metric_name : str
            the metric to get the log for (has to be one of the metric computed for this episode during training)

        Returns
        -------
        log : np.ndarray
            the log of metric values (empty if the metric was not computed during training)
        
        """
        metric_dict = defaultdict(list)
        log_file = os.path.join(self.project.project_path, "results", "logs", f"{self.episode}.txt")
        if not os.path.exists(log_file):
            return {}
        with open(log_file) as f:
            for line in f.readlines():
                if mode == "train" and line.startswith("[epoch"):
                    line = line.split("]: ")[1]
                elif mode == "val" and line.startswith("validation"):
                    line = line.split("validation: ")[1]
                else:
                    continue
                metrics = line.split(", ")
                for metric in metrics:
                    name, value = metric.split()
                    metric_dict[name].append(float(value))
        return metric_dict


    def update_plot_data(self):
        val_data = self.get_metric_log("val")
        if len(val_data) == 0:
            return
        for key, values in val_data.items():
            if key == "loss":
                continue
            if key not in self.val_data_lines:
                self.val_data_lines[key] = self.valGraph.plot(values, pen=pg.mkPen(color=self.colors[self.color_index], width=3), xRange=[0, 100], name=key)
                self.train_data_lines[key] = self.trainGraph.plot(values, pen=pg.mkPen(color=self.colors[self.color_index], width=3), xRange=[0, 100], name=key)
                self.color_index += 1
            values = val_data[key]
            self.val_data_lines[key].setData(range(len(values)), values)  # Update the data.

        train_data = self.get_metric_log("train")
        for key, values in train_data.items():
            if key == "loss":
                continue
            values = train_data[key]
            self.train_data_lines[key].setData(range(len(values)), values)

        if self.loss_data_lines == {}:
            self.loss_data_lines["val"] = self.lossGraph.plot([], pen=pg.mkPen(color="orange", width=3), name="validation", xRange=[0, 100])
            self.loss_data_lines["train"] = self.lossGraph.plot([], pen=pg.mkPen(color="blue", width=3), name="training", xRange=[0, 100])
        self.loss_data_lines["val"].setData(range(len(val_data["loss"])), val_data["loss"])
        self.loss_data_lines["train"].setData(range(len(train_data["loss"])), train_data["loss"])



def main():
    app = QApplication(sys.argv)

    project = Project("oft")
    window = EpisodeTraining(project, episode="test")
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()