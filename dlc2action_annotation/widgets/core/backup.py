#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy
# is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
""" Classes and methods handling data backups """
import threading
import warnings
from datetime import datetime
from pathlib import Path

from widgets.core.annotations import save_annotations
from widgets.viewer import Viewer


class RepeatingTimer(threading.Timer):
    """ Timer that repeats its function calls every N seconds """

    def run(self):
        """ Runs the timer """
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class BackupManager:
    """
    Attributes:
        backup_path: the folder where backups should be saved
        interval: the interval at which backups should be created, in minutes
        timer: the threading.Timer calling the save operation
    """

    def __init__(
        self,
        backup_path: Path,
        viewer: Viewer,
        interval: int,
    ):
        """
        Args:
            backup_path: the folder where backups should be saved
            viewer: the viewer object being used to label data
            interval: the interval at which backups should be created, in minutes
        """
        self.backup_path = backup_path
        self.interval = interval
        self._viewer = viewer
        self.timer = RepeatingTimer(60 * self.interval, self.create_backup)

        self.backup_path.mkdir(exist_ok=True, parents=True)
        self.create_backup()  # create initial backup

    def create_backup(self) -> None:
        """ Backs up the viewer data """
        now = datetime.now()
        day_str = now.strftime("%Y-%m-%d")
        output_folder = self.backup_path / day_str
        output_folder.mkdir(exist_ok=True)
        time_str = now.strftime("%Hh%Mm%Ss")
        output_file = output_folder / ("backup-" + time_str + ".pickle")
        metadata, cat_labels, animals, times = self._viewer.export_annotation_data()
        try:
            save_annotations(
                output_path=output_file,
                metadata=metadata,
                animals=animals,
                cat_labels=cat_labels,
                times=times,
                human_readable=True,
                overwrite=False,
            )
        except IOError as err:
            warnings.warn(f"Failed to back up data: {err}")
        finally:
            self.timer = threading.Timer(60 * self.interval, self.create_backup)

    def start(self) -> None:
        """ Starts the module, backing up data periodically """
        self.timer.start()

    def stop(self) -> None:
        """ Cancels the periodic backups """
        self.timer.cancel()
        self.timer = RepeatingTimer(60 * self.interval, self.create_backup)
