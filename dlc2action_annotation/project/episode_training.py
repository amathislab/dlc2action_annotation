from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtGui import QFont
from dlc2action.project import Project
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from time import sleep
from collections import defaultdict
import numpy as np


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, project, episode, episode_settings=None, load_search=None, *args, **kwargs):
        self.project = project
        self.episode = episode
        self.episode_settings = episode_settings
        self.load_search = load_search
        super().__init__(*args, **kwargs)

    def run(self):
        """Long-running task."""
        self.project.run_episode(
            self.episode,
            parameters_update=self.episode_settings,
            force=True,
            load_search=self.load_search,
        )
        self.finished.emit()


class EpisodeTraining(QtWidgets.QWidget):

    def __init__(self, project, episode, episode_settings=None, load_search=None, *args, **kwargs):
        super().__init__(*args, **kwargs)


        if episode_settings is None:
            episode_settings = project._read_parameters()
        self.project = project
        self.episode = episode
        self.episode_settings = episode_settings
        self.load_search = load_search
        
        self.graphWidget = pg.PlotWidget()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.graphWidget)

        self.graphWidget.setBackground('w')
        self.graphWidget.setXRange(0, self.episode_settings["training"]["num_epochs"], padding=0)
        self.graphWidget.setYRange(0, 1, padding=0)
        self.graphWidget.setLabel('bottom', 'Epoch', color='black', size="15pt")
        self.graphWidget.setTitle(f"episode {episode}", color='black', size="15pt")
        self.data_lines = {}
        legend = self.graphWidget.addLegend()
        legend.setLabelTextSize("15pt")

        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        self.color_index = 0
    
            # ... init continued ...
        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        self.run_episode()

    def run_episode(self):
        # Step 2: Create a QThread object
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
            lambda: print("Finished!")
        )


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
        data = self.get_metric_log("val")
        for key, values in data.items():
            if key not in self.data_lines:
                self.data_lines[key] = self.graphWidget.plot(values, pen=pg.mkPen(color=self.colors[self.color_index], width=3), xRange=[0, 100], name=key)
                self.color_index += 1
            values = data[key]
            self.data_lines[key].setData(range(len(values)), values)  # Update the data.



def main():
    app = QApplication(sys.argv)

    project = Project("oft")
    window = EpisodeTraining(project, episode="test")
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()