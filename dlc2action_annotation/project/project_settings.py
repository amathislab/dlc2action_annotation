import os
import sys
from pathlib import Path

from dlc2action.options import annotation_stores, input_stores
from dlc2action.project import Project
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dlc2action_annotation.project.utils import show_error, show_warning
from dlc2action_annotation.widgets.settings import (
    CategoryInputWidget,
    MultipleDoubleInputWidget,
    MultipleInputWidget,
)
from dlc2action_annotation.widgets.viewer import Viewer as Viewer


class NewCheckbox(QCheckBox):
    """Checkbox that returns False if unchecked, True if checked, and "???" if half-checked."""

    def isChecked(self):
        is_checked = super().checkState()
        value_dict = {0: False, 2: True, 1: "???"}
        return value_dict[is_checked]


class NewLineEdit(QLineEdit):
    """Line edit that returns None if empty, the text otherwise."""

    def text(self):
        text = super().text()
        if text == "None":
            return None
        return text


class TypeChoice(QWidget):
    """Widget for choosing the data and annotation types for a new project."""

    accepted = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        listBox = QVBoxLayout(self)
        self.setLayout(listBox)
        self.layout.setAlignment(Qt.AlignHCenter)
        self.data_combo = self.make_data_combo()
        self.annotation_combo = self.make_annotation_combo()
        self.data_label = QLabel("Data Type")
        self.annotation_label = QLabel("Annotation Type")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.labels = {}
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.data_combo)
        self.layout.addWidget(self.data_label)
        self.layout.addWidget(self.annotation_combo)
        self.layout.addWidget(self.annotation_label)
        self.layout.addWidget(self.buttonBox)
        scroll = QScrollArea(self)
        listBox.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)

        scrollContent.setLayout(self.layout)
        scroll.setWidget(scrollContent)

        self.data_changed()
        self.annotation_changed()

    def data_changed(self, index=None):
        self.data_type = self.data_combo.currentText()
        self.data_label.setText(input_stores[self.data_type].__doc__)

    def make_data_combo(self):
        combo = QComboBox()
        options = Project.data_types()
        for option in options:
            combo.addItem(option)
        combo.currentIndexChanged.connect(self.data_changed)
        return combo

    def make_annotation_combo(self):
        combo = QComboBox()
        options = Project.annotation_types()
        for option in options:
            combo.addItem(option)
        combo.currentIndexChanged.connect(self.annotation_changed)
        return combo

    def annotation_changed(self, index=None):
        self.annotation_type = self.annotation_combo.currentText()
        if self.annotation_type != "dlc":
            show_warning(
                "Annotation generation not supported.",
                "At the moment, only DLC annotations are supported directly; the other types would need to be converted manually.",
            )
        self.annotation_label.setText(annotation_stores[self.annotation_type].__doc__)

    def make_button(self):
        button = QPushButton("Create Project")
        button.clicked.connect(self.create_project)
        return button

    def accept(self):
        self.accepted.emit((self.data_type, self.annotation_type))
        self.close()

    def reject(self):
        print("Rejected")

    def sizeHint(self):
        return QSize(800, 500)


class ProjectSettings(QWidget):
    """Widget for editing `dlc2action` project settings."""

    accepted = pyqtSignal(dict)
    rejected = pyqtSignal()

    def __init__(
        self, settings, enabled=True, title=None, project=None, show_model=False
    ):
        super().__init__()
        self.settings = settings
        if (
            not isinstance(self.settings["data"].get("data_path", ""), str)
            and self.settings["data"]["data_path"] is not None
        ):
            show_error(
                "The interface does not support multiple data paths. Please edit the settings file manually."
            )
            return
        if (
            not isinstance(self.settings["data"].get("annotation_path", ""), str)
            and self.settings["data"]["annotation_path"] is not None
        ):
            show_error(
                "The interface does not support multiple annotation paths. Please edit the settings file manually."
            )
            return
        if (
            not isinstance(self.settings["data"].get("data_suffix", ""), str)
            and self.settings["data"]["data_suffix"] is not None
        ):
            show_error(
                "The interface does not support multiple data suffixes. Please edit the settings file manually."
            )
            return
        if (
            not isinstance(self.settings["data"].get("annotation_suffix", ""), str)
            and self.settings["data"]["annotation_suffix"] is not None
        ):
            show_error(
                "The interface does not support multiple annotation suffixes. Please edit the settings file manually."
            )
            return
        if (
            not isinstance(self.settings["data"].get("data_prefix", ""), str)
            and self.settings["data"]["data_prefix"] is not None
        ):
            show_error(
                "The interface does not support multiple data prefixes. Please edit the settings file manually."
            )
            return
        if (
            not isinstance(self.settings["data"].get("feature_suffix", ""), str)
            and self.settings["data"]["feature_suffix"] is not None
        ):
            show_error(
                "The interface does not support multiple feature suffixes. Please edit the settings file manually."
            )
            return
        self.enabled = enabled
        self.show_model = show_model
        self.project = project
        self.setWindowTitle(title)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.labels = {}
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.tabs = QTabWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.buttonBox)
        self.create_general_tab()
        self.set_general_tab()
        self.create_training_tab()
        self.set_training_tab()
        self.create_augmentations_tab()
        self.set_augmentations_tab()
        self.create_features_tab()
        self.set_features_tab()
        self.create_data_tab()
        self.set_data_tab()
        self.create_metrics_tab()
        self.set_metrics_tab()
        self.setLayout(self.layout)

    def accept(self):
        self.collect_general()
        self.collect_training()
        self.collect_augmentations()
        self.collect_features()
        self.collect_data()
        self.collect_metrics()
        blanks_ok = self.check_blanks(self.settings)
        if not blanks_ok:
            msg = QMessageBox()
            msg.setText("Please fill in all fields.")
            msg.setInformativeText(
                "The necessary fields are marked with ??? or half-checked checkboxes."
            )
            msg.exec_()
            return
        if self.project is not None:
            if (
                self.settings["general"]["model_name"]
                not in self.project.list_searches().index
            ):
                msg = QMessageBox()
                msg.setText("Hyperparameter search is not available.")
                msg.setInformativeText(
                    "There is no finished hyperparameter search for this model. Run the experiment with default parameters?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                if msg.exec_() == QMessageBox.No:
                    return
        self.accepted.emit(self.settings)
        self.close()

    def reject(self):
        self.rejected.emit()
        self.close()

    def create_tab(self, name):
        new_tab = QWidget()
        self.tabs.addTab(new_tab, name)
        scroll = QScrollArea()
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        new_tab.setLayout(layout)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)
        new_layout = QFormLayout()
        scrollContent.setLayout(new_layout)
        scroll.setWidget(scrollContent)
        return new_tab, new_layout

    def create_general_tab(self):
        self.general_tab, self.general_layout = self.create_tab("General")

    def create_training_tab(self):
        self.training_tab, self.training_layout = self.create_tab("Training")

    def create_augmentations_tab(self):
        self.augmentation_tab, self.augmentation_layout = self.create_tab(
            "Augmentations"
        )

    def create_features_tab(self):
        self.features_tab, self.features_layout = self.create_tab("Features")

    def create_data_tab(self):
        self.data_tab, self.data_layout = self.create_tab("Data")

    def create_metrics_tab(self):
        self.metrics_tab, self.metrics_layout = self.create_tab("Metrics")

    def set_general_tab(self):
        self.clearLayout(self.general_layout)
        self.set_general_tab_data()
        self.general_layout.addRow("Model name: ", self.model_name)
        self.general_layout.addRow("Allow only one action per frame: ", self.exclusive)
        self.general_layout.addRow("Only use annotated videos: ", self.only_annotated)
        self.general_layout.addRow("Agent names to ignore: ", self.ignored_agents)
        self.general_layout.addRow("Number of CPUs to use: ", self.num_cpus)
        self.general_layout.addRow("Overlap between segments: ", self.overlap)
        self.general_layout.addRow("Interactive mode: ", self.interactive)

    def set_training_tab(self):
        self.clearLayout(self.training_layout)
        self.set_training_tab_data()
        self.training_layout.addRow("Learning rate: ", self.lr)
        self.training_layout.addRow("Weight decay: ", self.weight_decay)
        self.training_layout.addRow("Device (auto, cpu, cuda:0, ...): ", self.device)
        self.training_layout.addRow(
            "Train on multiple devices in parallel: ", self.parallel
        )
        self.training_layout.addRow(
            "Validate every n epochs: ", self.validation_interval
        )
        self.training_layout.addRow(
            "Save model every n epochs: ", self.model_save_epochs
        )
        self.training_layout.addRow("Number of epochs: ", self.num_epochs)
        self.training_layout.addRow("Batch size: ", self.batch_size)
        self.training_layout.addRow("Normalize data: ", self.normalize)
        self.training_layout.addRow(
            "Temporal subsampling fraction: ", self.temporal_subsampling_size
        )
        self.training_layout.addRow("Partition method: ", self.partition_method)
        self.training_layout.addRow("Validation fraction: ", self.val_frac)
        self.training_layout.addRow("Test fraction: ", self.test_frac)

    def set_augmentations_tab(self):
        self.clearLayout(self.augmentation_layout)
        self.set_augmentations_tab_data()
        self.augmentation_layout.addRow("Augmentations: ", self.augmentations)
        self.augmentation_layout.addRow("Rotation limit (rad): ", self.rotation_limit)
        self.augmentation_layout.addRow("Mirror dimensions: ", self.mirror_dim)
        self.augmentation_layout.addRow("Noise standard deviation: ", self.noise_std)
        self.augmentation_layout.addRow("Zoom limits: ", self.zoom_limits)

    def set_features_tab(self):
        self.clearLayout(self.features_layout)
        self.set_features_tab_data()
        self.features_layout.addRow("Features: ", self.keys)
        self.features_layout.addRow("Distance pairs: ", self.distance_pairs)

    def set_data_tab(self):
        self.clearLayout(self.data_layout)
        self.set_data_tab_data()
        if "behaviors" in self.settings["data"]:
            self.data_layout.addRow("Behaviors: ", self.behaviors)
        if "ignored_classes" in self.settings["data"]:
            self.data_layout.addRow("Ignored classes: ", self.ignored_classes)
        if "correction" in self.settings["data"]:
            self.data_layout.addRow("Correction: ", self.correction)
        if "error_class" in self.settings["data"]:
            self.data_layout.addRow("Error class: ", self.error_class)
        if "min_frames_action" in self.settings["data"]:
            self.data_layout.addRow(
                "Minimum frames per action: ", self.min_frames_action
            )
        if "filter_annotated" in self.settings["data"]:
            self.data_layout.addRow("Filter annotated: ", self.filter_annotated)
        if "filter_background" in self.settings["data"]:
            self.data_layout.addRow("Filter background: ", self.filter_background)
        if "visibility_min_score" in self.settings["data"]:
            self.data_layout.addRow("Visibility min score: ", self.visibility_min_score)
        if "visibility_min_frac" in self.settings["data"]:
            self.data_layout.addRow(
                "Visibility min fraction: ", self.visibility_min_frac
            )
        if "annotation_suffix" in self.settings["data"]:
            self.data_layout.addRow("Annotation suffix: ", self.annotation_suffix)
        if "use_hard_negatives" in self.settings["data"]:
            self.data_layout.addRow("Use hard negatives: ", self.use_hard_negatives)
        if "separator" in self.settings["data"]:
            self.data_layout.addRow("Separator: ", self.separator)
        if "treba_files" in self.settings["data"]:
            self.data_layout.addRow("Use treba files: ", self.treba_files)
        if "task_n" in self.settings["data"]:
            self.data_layout.addRow("Task number: ", self.task_n)
        if "tracking_suffix" in self.settings["data"]:
            self.data_layout.addRow("Tracking suffix: ", self.tracking_suffix)
        if "tracking_path" in self.settings["data"]:
            self.data_layout.addRow("Tracking path: ", self.tracking_path)
        if "frame_limit" in self.settings["data"]:
            self.data_layout.addRow("Frame limit: ", self.frame_limit)
        if "default_agent_name" in self.settings["data"]:
            self.data_layout.addRow("Default agent name: ", self.default_agent_name)
        if "data_suffix" in self.settings["data"]:
            self.data_layout.addRow("Data suffix: ", self.data_suffix)
        if "data_prefix" in self.settings["data"]:
            self.data_layout.addRow("Data prefix: ", self.data_prefix)
        if "feature_suffix" in self.settings["data"]:
            self.data_layout.addRow("Feature suffix: ", self.feature_suffix)
        if "convert_int_indices" in self.settings["data"]:
            self.data_layout.addRow("Convert int indices: ", self.convert_int_indices)
        if "canvas_shape" in self.settings["data"]:
            self.data_layout.addRow("Canvas shape: ", self.canvas_shape)
        if "ignored_bodyparts" in self.settings["data"]:
            self.data_layout.addRow("Ignored bodyparts: ", self.ignored_bodyparts)
        if "likelihood_threshold" in self.settings["data"]:
            self.data_layout.addRow("Likelihood threshold: ", self.likelihood_threshold)
        if "centered" in self.settings["data"]:
            self.data_layout.addRow("Centered: ", self.centered)
        if "use_features" in self.settings["data"]:
            self.data_layout.addRow("Use features: ", self.use_features)
        if "behavior_file" in self.settings["data"]:
            self.data_layout.addRow("Behavior file: ", self.behavior_file)
        if "fps" in self.settings["data"]:
            self.data_layout.addRow("FPS: ", self.fps)

    def set_metrics_tab(self):
        self.clearLayout(self.metrics_layout)
        self.set_metrics_tab_data()
        label = QLabel("Recall:")
        label.setStyleSheet("font-weight: bold;")
        self.metrics_layout.addRow(label)
        self.metrics_layout.addRow("Average: ", self.recall_average)
        self.metrics_layout.addRow("Ignored classes: ", self.recall_ignored_classes)
        self.metrics_layout.addRow("Tag average: ", self.recall_tag_average)
        self.metrics_layout.addRow("Threshold value: ", self.recall_threshold_value)
        label = QLabel("\nPrecision:")
        label.setStyleSheet("font-weight: bold;")
        self.metrics_layout.addRow(label)
        self.metrics_layout.addRow("Average: ", self.precision_average)
        self.metrics_layout.addRow("Ignored classes: ", self.precision_ignored_classes)
        self.metrics_layout.addRow("Tag average: ", self.precision_tag_average)
        self.metrics_layout.addRow("Threshold value: ", self.precision_threshold_value)
        label = QLabel("\nF1:")
        label.setStyleSheet("font-weight: bold;")
        self.metrics_layout.addRow(label)
        self.metrics_layout.addRow("Average: ", self.f1_average)
        self.metrics_layout.addRow("Ignored classes: ", self.f1_ignored_classes)
        self.metrics_layout.addRow("Tag average: ", self.f1_tag_average)
        self.metrics_layout.addRow("Threshold value: ", self.f1_threshold_value)
        # label = QLabel("\nF beta:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Beta: ", self.f_beta_beta)
        # self.metrics_layout.addRow("Average: ", self.f_beta_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.f_beta_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.f_beta_tag_average)
        # self.metrics_layout.addRow("Threshold value: ", self.f_beta_threshold_value)
        # label = QLabel("\nCount:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Classes: ", self.count_classes)
        # label = QLabel("\nSegmental precision:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Average: ", self.segmental_precision_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.segmental_precision_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.segmental_precision_tag_average)
        # self.metrics_layout.addRow("Threshold value: ", self.segmental_precision_threshold_value)
        # label = QLabel("\nSegmental recall:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Average: ", self.segmental_recall_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.segmental_recall_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.segmental_recall_tag_average)
        # self.metrics_layout.addRow("Threshold value: ", self.segmental_recall_threshold_value)
        # label = QLabel("\nSegmental F1:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Average: ", self.segmental_f1_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.segmental_f1_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.segmental_f1_tag_average)
        # self.metrics_layout.addRow("Threshold value: ", self.segmental_f1_threshold_value)
        # label = QLabel("\nSegmental F beta:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Beta: ", self.segmental_f_beta_beta)
        # self.metrics_layout.addRow("Average: ", self.segmental_f_beta_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.segmental_f_beta_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.segmental_f_beta_tag_average)
        # self.metrics_layout.addRow("Threshold value: ", self.segmental_f_beta_threshold_value)
        # label = QLabel("\PR-AUC:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Average: ", self.pr_auc_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.pr_auc_ignored_classes)
        # self.metrics_layout.addRow("Tag average: ", self.pr_auc_tag_average)
        # self.metrics_layout.addRow("Threshold step: ", self.pr_auc_threshold_step)
        # label = QLabel("\mAP:")
        # label.setStyleSheet("font-weight: bold;")
        # self.metrics_layout.addRow(label)
        # self.metrics_layout.addRow("Average: ", self.map_average)
        # self.metrics_layout.addRow("Ignored classes: ", self.map_ignored_classes)
        # self.metrics_layout.addRow("IoU threshold: ", self.map_iou_threshold)
        # self.metrics_layout.addRow("Threshold value: ", self.map_threshold_value)

    def set_general_tab_data(self):
        self.model_name = self.set_combo(
            "general",
            "model_name",
            ["mlp", "ms_tcn3", "c2f_tcn", "c2f_transformer", "asformer", "transformer"],
        )
        self.exclusive = self.set_toggle("general", "exclusive")
        self.only_annotated = self.set_toggle("general", "only_load_annotated")
        self.ignored_agents = self.set_multiple_input("general", "ignored_clips")
        self.num_cpus = self.set_le("general", "num_cpus", set_int=True)
        self.overlap = self.set_slider("general", "overlap", 0, 1, percent=True)
        self.interactive = self.set_toggle("general", "interactive")

    def set_training_tab_data(self):
        self.lr = self.set_le("training", "lr", set_int=False, set_float=True)
        self.weight_decay = self.set_le(
            "training", "weight_decay", set_int=False, set_float=True
        )
        self.device = self.set_le("training", "device", set_int=False, set_float=False)
        self.validation_interval = self.set_le(
            "training", "validation_interval", set_int=True, set_float=False
        )
        self.num_epochs = self.set_le(
            "training", "num_epochs", set_int=True, set_float=False
        )
        self.batch_size = self.set_le(
            "training", "batch_size", set_int=True, set_float=False
        )
        self.model_save_epochs = self.set_le(
            "training", "model_save_epochs", set_int=True, set_float=False
        )
        self.normalize = self.set_toggle("training", "normalize")
        self.parallel = self.set_toggle("training", "parallel")
        self.temporal_subsampling_size = self.set_slider(
            "training", "temporal_subsampling_size", 0, 1, percent=True
        )
        self.partition_method = self.set_combo(
            "training",
            "partition_method",
            [
                "random",
                "file",
                "random:equalize:segments",
                "random:equalize:videos",
                "folders",
                "time",
                "time:strict",
            ],
        )
        self.val_frac = self.set_slider("training", "val_frac", 0, 1, percent=True)
        self.test_frac = self.set_slider("training", "test_frac", 0, 1, percent=True)
        self.split_path = self.set_file("training", "split_path", dir=False)

    def set_augmentations_tab_data(self):
        self.augmentations = self.set_options(
            "augmentations",
            "augmentations",
            ["rotate", "real_lens", "add_noise", "shift", "zoom", "mirror", "switch"],
        )
        self.rotation_limit = self.set_limits("augmentations", "rotation_limits")
        self.mirror_dim = self.set_options("augmentations", "mirror_dim", [0, 1, 2])
        self.noise_std = self.set_le(
            "augmentations", "noise_std", set_int=False, set_float=True
        )
        self.zoom_limits = self.set_limits("augmentations", "zoom_limits")

    def set_features_tab_data(self):
        self.keys = self.set_options(
            "features",
            "keys",
            [
                "coords",
                "coord_diff",
                "center",
                "intra_distance",
                "inter_distance",
                "speed_direction",
                "speed_value",
                "acc_joints",
                "likelihood",
            ],
        )
        self.distance_pairs = self.set_multiple_input(
            "features", "distance_pairs", type="double"
        )
        # not including the rest of the features because writing the input fields is a pain

    def set_data_tab_data(self):
        if "behaviors" in self.settings["data"]:
            self.behaviors = self.set_multiple_input("data", "behaviors")
        if "ignored_classes" in self.settings["data"]:
            self.ignored_classes = self.set_multiple_input("data", "ignored_classes")
        if "correction" in self.settings["data"]:
            self.correction = self.set_multiple_input(
                "data", "correction", type="double"
            )
        if "error_class" in self.settings["data"]:
            self.error_class = self.set_le(
                "data", "error_class", set_int=False, set_float=False
            )
        if "min_frames_action" in self.settings["data"]:
            self.min_frames_action = self.set_le(
                "data", "min_frames_action", set_int=True, set_float=False
            )
        if "filter_annotated" in self.settings["data"]:
            self.filter_annotated = self.set_toggle("data", "filter_annotated")
        if "filter_background" in self.settings["data"]:
            self.filter_background = self.set_toggle("data", "filter_background")
        if "visibility_min_score" in self.settings["data"]:
            self.visibility_min_score = self.set_slider(
                "data", "visibility_min_score", 0, 1, percent=True
            )
        if "visibility_min_frac" in self.settings["data"]:
            self.visibility_min_frac = self.set_slider(
                "data", "visibility_min_frac", 0, 1, percent=True
            )
        if "annotation_suffix" in self.settings["data"]:
            self.annotation_suffix = self.set_le(
                "data", "annotation_suffix", set_int=False, set_float=False
            )
        if "use_hard_negatives" in self.settings["data"]:
            self.use_hard_negatives = self.set_toggle("data", "use_hard_negatives")
        if "separator" in self.settings["data"]:
            self.separator = self.set_le(
                "data", "separator", set_int=False, set_float=False
            )
        if "treba_files" in self.settings["data"]:
            self.treba_files = self.set_toggle("data", "treba_files")
        if "task_n" in self.settings["data"]:
            self.task_n = self.set_combo("data", "task_n", [1, 2])
        if "tracking_suffix" in self.settings["data"]:
            self.tracking_suffix = self.set_le(
                "data", "tracking_suffix", set_int=False, set_float=False
            )
        if "tracking_path" in self.settings["data"]:
            self.tracking_path = self.set_file("data", "tracking_path", dir=True)
        if "frame_limit" in self.settings["data"]:
            self.frame_limit = self.set_le(
                "data", "frame_limit", set_int=True, set_float=False
            )
        if "default_agent_name" in self.settings["data"]:
            self.default_agent_name = self.set_le(
                "data", "default_agent_name", set_int=False, set_float=False
            )
        if "data_suffix" in self.settings["data"]:
            self.data_suffix = self.set_le(
                "data", "data_suffix", set_int=False, set_float=False
            )
        if "data_prefix" in self.settings["data"]:
            self.data_prefix = self.set_le(
                "data", "data_prefix", set_int=False, set_float=False
            )
        if "feature_suffix" in self.settings["data"]:
            self.feature_suffix = self.set_le(
                "data", "feature_suffix", set_int=False, set_float=False
            )
        if "convert_int_indices" in self.settings["data"]:
            self.convert_int_indices = self.set_toggle("data", "convert_int_indices")
        if "canvas_shape" in self.settings["data"]:
            self.canvas_shape = self.set_dimensions(
                "data",
                "canvas_shape",
                num_fields=3 if self.settings["general"]["data_type"] == "np_3d" else 2,
            )
        if "ignored_bodyparts" in self.settings["data"]:
            self.ignored_bodyparts = self.set_multiple_input(
                "data", "ignored_bodyparts"
            )
        if "likelihood_threshold" in self.settings["data"]:
            self.likelihood_threshold = self.set_le(
                "data", "likelihood_threshold", set_int=False, set_float=True
            )
        if "centered" in self.settings["data"]:
            self.centered = self.set_toggle("data", "centered")
        if "use_features" in self.settings["data"]:
            self.use_features = self.set_toggle("data", "use_features")
        if "behavior_file" in self.settings["data"]:
            self.behavior_file = self.set_file("data", "behavior_file", dir=False)
        if "fps" in self.settings["data"]:
            self.fps = self.set_le("data", "fps", set_int=False, set_float=True)
        if "annotation_suffix" in self.settings["data"]:
            self.annotation_suffix = self.set_le(
                "data", "annotation_suffix", set_int=False, set_float=False
            )
        if "ignored_classes" in self.settings["data"]:
            self.ignored_classes = self.set_multiple_input("data", "ignored_classes")
        if "correction" in self.settings["data"]:
            self.correction = self.set_multiple_input(
                "data", "correction", type="double"
            )

    def set_metrics_tab_data(self):
        self.recall_average = self.set_combo(
            "metrics", "average", ["micro", "macro", "none"], subcategory="recall"
        )
        self.recall_ignored_classes = self.set_multiple_input(
            "metrics", "ignored_classes", subcategory="recall"
        )
        self.recall_tag_average = self.set_combo(
            "metrics", "tag_average", ["micro", "macro", "none"], subcategory="recall"
        )
        self.recall_threshold_value = self.set_slider(
            "metrics", "threshold_value", 0, 1, percent=True, subcategory="recall"
        )
        self.precision_average = self.set_combo(
            "metrics", "average", ["micro", "macro", "none"], subcategory="precision"
        )
        self.precision_ignored_classes = self.set_multiple_input(
            "metrics", "ignored_classes", subcategory="precision"
        )
        self.precision_tag_average = self.set_combo(
            "metrics",
            "tag_average",
            ["micro", "macro", "none"],
            subcategory="precision",
        )
        self.precision_threshold_value = self.set_slider(
            "metrics", "threshold_value", 0, 1, percent=True, subcategory="precision"
        )
        self.f1_average = self.set_combo(
            "metrics", "average", ["micro", "macro", "none"], subcategory="f1"
        )
        self.f1_ignored_classes = self.set_multiple_input(
            "metrics", "ignored_classes", subcategory="f1"
        )
        self.f1_tag_average = self.set_combo(
            "metrics", "tag_average", ["micro", "macro", "none"], subcategory="f1"
        )
        self.f1_threshold_value = self.set_slider(
            "metrics", "threshold_value", 0, 1, percent=True, subcategory="f1"
        )
        # self.f_beta_beta = self.set_le("metrics", "beta", set_int=False, set_float=True, subcategory="f_beta")
        # self.f_beta_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="f_beta")
        # self.f_beta_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="f_beta")
        # self.f_beta_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="f_beta")
        # self.f_beta_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="f_beta")
        # self.count_classes = self.set_multiple_input("metrics", "classes", subcategory="count")
        # self.segmental_precision_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="segmental_precision")
        # self.segmental_precision_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="segmental_precision")
        # self.segmental_precision_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="segmental_precision")
        # self.segmental_precision_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="segmental_precision")
        # self.segmental_recall_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="segmental_recall")
        # self.segmental_recall_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="segmental_recall")
        # self.segmental_recall_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="segmental_recall")
        # self.segmental_recall_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="segmental_recall")
        # self.segmental_f1_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="segmental_f1")
        # self.segmental_f1_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="segmental_f1")
        # self.segmental_f1_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="segmental_f1")
        # self.segmental_f1_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="segmental_f1")
        # self.segmental_f_beta_beta = self.set_le("metrics", "beta", set_int=False, set_float=True, subcategory="segmental_f_beta")
        # self.segmental_f_beta_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="segmental_f_beta")
        # self.segmental_f_beta_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="segmental_f_beta")
        # self.segmental_f_beta_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="segmental_f_beta")
        # self.segmental_f_beta_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="segmental_f_beta")
        # self.pr_auc_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="pr-auc")
        # self.pr_auc_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="pr-auc")
        # self.pr_auc_tag_average = self.set_combo("metrics", "tag_average", ["micro", "macro", "none"], subcategory="pr-auc")
        # self.pr_auc_threshold_step = self.set_slider("metrics", "threshold_step", 0, 1, percent=True, subcategory="pr-auc")
        # self.map_average = self.set_combo("metrics", "average", ["micro", "macro", "none"], subcategory="mAP")
        # self.map_ignored_classes = self.set_multiple_input("metrics", "ignored_classes", subcategory="mAP")
        # self.map_iou_threshold = self.set_slider("metrics", "iou_threshold", 0, 1, percent=True, subcategory="mAP")
        # self.map_threshold_value = self.set_slider("metrics", "threshold_value", 0, 1, percent=True, subcategory="mAP")

    def collect_general(self):
        self.settings["general"]["model_name"] = self.model_name.currentText()
        self.settings["general"]["exclusive"] = self.exclusive.isChecked()
        self.settings["general"][
            "only_load_annotated"
        ] = self.only_annotated.isChecked()
        self.settings["general"]["ignored_clips"] = self.ignored_agents.values()
        self.settings["general"]["num_cpus"] = int(self.num_cpus.text())
        self.settings["general"]["overlap"] = (
            self.overlap.itemAt(0).widget().value() / 100
        )
        self.settings["general"]["interactive"] = self.interactive.isChecked()

    def collect_training(self):
        self.settings["training"]["lr"] = float(self.lr.text())
        self.settings["training"]["weight_decay"] = float(self.weight_decay.text())
        self.settings["training"]["device"] = self.device.text()
        self.settings["training"]["validation_interval"] = int(
            self.validation_interval.text()
        )
        self.settings["training"]["num_epochs"] = int(self.num_epochs.text())
        self.settings["training"]["batch_size"] = int(self.batch_size.text())
        self.settings["training"]["model_save_epochs"] = int(
            self.model_save_epochs.text()
        )
        self.settings["training"]["normalize"] = self.normalize.isChecked()
        self.settings["training"]["parallel"] = self.parallel.isChecked()
        self.settings["training"]["temporal_subsampling_size"] = (
            self.temporal_subsampling_size.itemAt(0).widget().value() / 100
        )
        self.settings["training"][
            "partition_method"
        ] = self.partition_method.currentText()
        self.settings["training"]["val_frac"] = (
            self.val_frac.itemAt(0).widget().value() / 100
        )
        self.settings["training"]["test_frac"] = (
            self.test_frac.itemAt(0).widget().value() / 100
        )
        self.settings["training"]["split_path"] = (
            self.split_path.itemAt(0).widget().text()
        )

    def collect_augmentations(self):
        self.settings["augmentations"]["augmentations"] = self.collect_options(
            self.augmentations
        )
        self.settings["augmentations"]["rotation_limits"] = self.collect_limits(
            self.rotation_limit
        )
        self.settings["augmentations"]["mirror_dim"] = {
            int(x) for x in self.collect_options(self.mirror_dim)
        }
        self.settings["augmentations"]["noise_std"] = float(self.noise_std.text())
        self.settings["augmentations"]["zoom_limits"] = self.collect_limits(
            self.zoom_limits
        )

    def collect_options(self, options_layout):
        return set(
            [
                options_layout.itemAt(i).widget().text()
                for i in range(options_layout.count())
                if options_layout.itemAt(i).widget().isChecked()
            ]
        )

    def collect_limits(self, limits_layout):
        return [
            float(limits_layout.itemAt(i).widget().text())
            for i in range(limits_layout.count())
            if isinstance(limits_layout.itemAt(i).widget(), QLineEdit)
        ]

    def collect_features(self):
        self.settings["features"]["keys"] = self.collect_options(self.keys)
        self.settings["features"]["distance_pairs"] = self.distance_pairs.values()

    def collect_data(self):
        if "behaviors" in self.settings["data"]:
            self.settings["data"]["behaviors"] = self.behaviors.values()
        if "ignored_classes" in self.settings["data"]:
            self.settings["data"]["ignored_classes"] = self.ignored_classes.values()
        if "correction" in self.settings["data"]:
            self.settings["data"]["correction"] = self.correction.values()
        if "error_class" in self.settings["data"]:
            self.settings["data"]["error_class"] = self.error_class.text()
        if "min_frames_action" in self.settings["data"]:
            self.settings["data"]["min_frames_action"] = int(
                self.min_frames_action.text()
            )
        if "filter_annotated" in self.settings["data"]:
            self.settings["data"][
                "filter_annotated"
            ] = self.filter_annotated.isChecked()
        if "filter_background" in self.settings["data"]:
            self.settings["data"][
                "filter_background"
            ] = self.filter_background.isChecked()
        if "visibility_min_score" in self.settings["data"]:
            self.settings["data"]["visibility_min_score"] = (
                self.visibility_min_score.itemAt(0).widget().value() / 100
            )
        if "visibility_min_frac" in self.settings["data"]:
            self.settings["data"]["visibility_min_frac"] = (
                self.visibility_min_frac.itemAt(0).widget().value() / 100
            )
        if "annotation_suffix" in self.settings["data"]:
            self.settings["data"]["annotation_suffix"] = self.annotation_suffix.text()
        if "use_hard_negatives" in self.settings["data"]:
            self.settings["data"][
                "use_hard_negatives"
            ] = self.use_hard_negatives.isChecked()
        if "separator" in self.settings["data"]:
            self.settings["data"]["separator"] = self.separator.text()
        if "treba_files" in self.settings["data"]:
            self.settings["data"]["treba_files"] = self.treba_files.isChecked()
        if "task_n" in self.settings["data"]:
            self.settings["data"]["task_n"] = int(self.task_n.currentText())
        if "tracking_suffix" in self.settings["data"]:
            self.settings["data"]["tracking_suffix"] = self.tracking_suffix.text()
        if "tracking_path" in self.settings["data"]:
            self.settings["data"]["tracking_path"] = (
                self.tracking_path.itemAt(0).widget().text()
            )
        if "frame_limit" in self.settings["data"]:
            self.settings["data"]["frame_limit"] = int(self.frame_limit.text())
        if "default_agent_name" in self.settings["data"]:
            self.settings["data"]["default_agent_name"] = self.default_agent_name.text()
        if "data_suffix" in self.settings["data"]:
            self.settings["data"]["data_suffix"] = self.data_suffix.text()
        if "data_prefix" in self.settings["data"]:
            self.settings["data"]["data_prefix"] = self.data_prefix.text()
        if "feature_suffix" in self.settings["data"]:
            self.settings["data"]["feature_suffix"] = self.feature_suffix.text()
        if "convert_int_indices" in self.settings["data"]:
            self.settings["data"][
                "convert_int_indices"
            ] = self.convert_int_indices.isChecked()
        if "canvas_shape" in self.settings["data"]:
            texts = [
                self.canvas_shape.itemAt(i).widget().text()
                for i in range(self.canvas_shape.count())
            ]
            if any([x == "???" for x in texts]):
                self.settings["data"]["canvas_shape"] = "???"
            else:
                self.settings["data"]["canvas_shape"] = [int(x) for x in texts]
        if "ignored_bodyparts" in self.settings["data"]:
            self.settings["data"]["ignored_bodyparts"] = self.ignored_bodyparts.values()
        if "likelihood_threshold" in self.settings["data"]:
            self.settings["data"]["likelihood_threshold"] = float(
                self.likelihood_threshold.text()
            )
        if "centered" in self.settings["data"]:
            self.settings["data"]["centered"] = self.centered.isChecked()
        if "use_features" in self.settings["data"]:
            self.settings["data"]["use_features"] = self.use_features.isChecked()
        if "behavior_file" in self.settings["data"]:
            self.settings["data"]["behavior_file"] = (
                self.behavior_file.itemAt(0).widget().text()
            )
        if "fps" in self.settings["data"]:
            self.settings["data"]["fps"] = float(self.fps.text())
        if "annotation_suffix" in self.settings["data"]:
            self.settings["data"]["annotation_suffix"] = self.annotation_suffix.text()
        if "ignored_classes" in self.settings["data"]:
            self.settings["data"]["ignored_classes"] = self.ignored_classes.values()
        if "correction" in self.settings["data"]:
            self.settings["data"]["correction"] = {
                k: v for k, v in self.correction.values()
            }

    def collect_metrics(self):
        self.settings["metrics"]["recall"][
            "average"
        ] = self.recall_average.currentText()
        self.settings["metrics"]["recall"][
            "ignored_classes"
        ] = self.recall_ignored_classes.values()
        self.settings["metrics"]["recall"][
            "tag_average"
        ] = self.recall_tag_average.currentText()
        self.settings["metrics"]["recall"]["threshold_value"] = (
            self.recall_threshold_value.itemAt(0).widget().value() / 100
        )
        self.settings["metrics"]["precision"][
            "average"
        ] = self.precision_average.currentText()
        self.settings["metrics"]["precision"][
            "ignored_classes"
        ] = self.precision_ignored_classes.values()
        self.settings["metrics"]["precision"][
            "tag_average"
        ] = self.precision_tag_average.currentText()
        self.settings["metrics"]["precision"]["threshold_value"] = (
            self.precision_threshold_value.itemAt(0).widget().value() / 100
        )
        self.settings["metrics"]["f1"]["average"] = self.f1_average.currentText()
        self.settings["metrics"]["f1"][
            "ignored_classes"
        ] = self.f1_ignored_classes.values()
        self.settings["metrics"]["f1"][
            "tag_average"
        ] = self.f1_tag_average.currentText()
        self.settings["metrics"]["f1"]["threshold_value"] = (
            self.f1_threshold_value.itemAt(0).widget().value() / 100
        )
        # self.settings["metrics"]["f_beta"]["beta"] = float(self.f_beta_beta.text())
        # self.settings["metrics"]["f_beta"]["average"] = self.f_beta_average.currentText()
        # self.settings["metrics"]["f_beta"]["ignored_classes"] = self.f_beta_ignored_classes.values()
        # self.settings["metrics"]["f_beta"]["tag_average"] = self.f_beta_tag_average.currentText()
        # self.settings["metrics"]["f_beta"]["threshold_value"] = self.f_beta_threshold_value.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["count"]["classes"] = self.count_classes.values()
        # self.settings["metrics"]["segmental_precision"]["average"] = self.segmental_precision_average.currentText()
        # self.settings["metrics"]["segmental_precision"]["ignored_classes"] = self.segmental_precision_ignored_classes.values()
        # self.settings["metrics"]["segmental_precision"]["tag_average"] = self.segmental_precision_tag_average.currentText()
        # self.settings["metrics"]["segmental_precision"]["threshold_value"] = self.segmental_precision_threshold_value.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["segmental_recall"]["average"] = self.segmental_recall_average.currentText()
        # self.settings["metrics"]["segmental_recall"]["ignored_classes"] = self.segmental_recall_ignored_classes.values()
        # self.settings["metrics"]["segmental_recall"]["tag_average"] = self.segmental_recall_tag_average.currentText()
        # self.settings["metrics"]["segmental_recall"]["threshold_value"] = self.segmental_recall_threshold_value.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["segmental_f1"]["average"] = self.segmental_f1_average.currentText()
        # self.settings["metrics"]["segmental_f1"]["ignored_classes"] = self.segmental_f1_ignored_classes.values()
        # self.settings["metrics"]["segmental_f1"]["tag_average"] = self.segmental_f1_tag_average.currentText()
        # self.settings["metrics"]["segmental_f1"]["threshold_value"] = self.segmental_f1_threshold_value.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["segmental_f_beta"]["beta"] = float(self.segmental_f_beta_beta.text())
        # self.settings["metrics"]["segmental_f_beta"]["average"] = self.segmental_f_beta_average.currentText()
        # self.settings["metrics"]["segmental_f_beta"]["ignored_classes"] = self.segmental_f_beta_ignored_classes.values()
        # self.settings["metrics"]["segmental_f_beta"]["tag_average"] = self.segmental_f_beta_tag_average.currentText()
        # self.settings["metrics"]["segmental_f_beta"]["threshold_value"] = self.segmental_f_beta_threshold_value.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["pr-auc"]["average"] = self.pr_auc_average.currentText()
        # self.settings["metrics"]["pr-auc"]["ignored_classes"] = self.pr_auc_ignored_classes.values()
        # self.settings["metrics"]["pr-auc"]["tag_average"] = self.pr_auc_tag_average.currentText()
        # self.settings["metrics"]["pr-auc"]["threshold_step"] = self.pr_auc_threshold_step.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["mAP"]["average"] = self.map_average.currentText()
        # self.settings["metrics"]["mAP"]["ignored_classes"] = self.map_ignored_classes.values()
        # self.settings["metrics"]["mAP"]["iou_threshold"] = self.map_iou_threshold.itemAt(0).widget().value() / 100
        # self.settings["metrics"]["mAP"]["threshold_value"] = self.map_threshold_value.itemAt(0).widget().value() / 100

    def get_value(self, category, field, subcategory=None):
        if subcategory is None:
            return self.settings[category][field]
        else:
            return self.settings[category][subcategory][field]

    def set_toggle(self, category, field, subcategory=None):
        toggle = NewCheckbox()
        value = self.get_value(category, field, subcategory)
        if value != "???":
            toggle.setChecked(value)
        else:
            toggle.tristate = True
            toggle.setCheckState(Qt.PartiallyChecked)
        toggle.setEnabled(self.enabled)
        return toggle

    def set_multiple_input(self, category, field, type="single", subcategory=None):
        value = self.get_value(category, field, subcategory)
        if value is None:
            if type == "category":
                x = {}
            else:
                x = []
        else:
            x = value
        if type == "double":
            widget = MultipleDoubleInputWidget(x)
        elif type == "category":
            widget = CategoryInputWidget(x)
        else:
            widget = MultipleInputWidget(x)
        widget.setEnabled(self.enabled)
        return widget

    def set_le(self, category, field, set_int=True, set_float=False, subcategory=None):
        le = NewLineEdit()
        if set_int:
            le.setValidator(QIntValidator())
        elif set_float:
            le.setValidator(QDoubleValidator())
        le.setText(str(self.get_value(category, field, subcategory=subcategory)))
        le.setEnabled(self.enabled)
        return le

    def set_slider(
        self, category, field, minimum, maximum, percent=False, subcategory=None
    ):
        slider_layout = QHBoxLayout()
        value = self.get_value(category, field, subcategory=subcategory)
        if percent:
            minimum *= 100
            maximum *= 100
            value *= 100
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        label = QLabel(str(value))
        slider.valueChanged.connect(lambda: self.update_label(slider, label))
        slider_layout.addWidget(slider)
        slider_layout.addWidget(label)
        slider.setEnabled(self.enabled)
        return slider_layout

    def set_combo(self, category, field, options, subcategory=None):
        combo = QComboBox()
        for o in options:
            combo.addItem(str(o))
        combo.setCurrentIndex(
            options.index(self.get_value(category, field, subcategory=subcategory))
        )
        combo.setEnabled(self.enabled)
        return combo

    def set_file(self, category, field, filter=None, dir=False, subcategory=None):
        layout = QHBoxLayout()
        file = self.get_value(category, field, subcategory=subcategory)
        file = file if file is not None else "None"
        button = QPushButton("Find")
        label = QLabel(os.path.basename(file))
        if dir:
            button.clicked.connect(lambda: self.get_dir(label, category, field, filter))
        else:
            button.clicked.connect(
                lambda: self.get_file(label, category, field, filter)
            )
        layout.addWidget(label)
        if self.enabled:
            layout.addWidget(button)
        return layout

    def get_file(self, label_widget, category, field, filter=None):
        file = QFileDialog().getOpenFileName(self, filter=filter)[0]
        label_widget.setText(os.path.basename(file))
        self.settings[category][field] = file

    def set_options(self, category, field, options, subcategory=None):
        layout = QFormLayout()
        settings_options = self.get_value(category, field, subcategory) or []
        for option in options:
            toggle = QCheckBox(str(option))
            toggle.setChecked(option in settings_options)
            toggle.setEnabled(self.enabled)
            layout.addRow(toggle)
        return layout

    def set_limits(self, category, field, subcategory=None):
        layout = QHBoxLayout()
        min_value, max_value = self.get_value(category, field, subcategory)
        layout.addWidget(QLabel("Min: "))
        le = QLineEdit()
        le.setValidator(QDoubleValidator())
        le.setText(str(min_value))
        le.setEnabled(self.enabled)
        layout.addWidget(le)
        layout.addWidget(QLabel("Max: "))
        le = QLineEdit()
        le.setValidator(QDoubleValidator())
        le.setText(str(max_value))
        le.setEnabled(self.enabled)
        layout.addWidget(le)
        return layout

    def set_dimensions(self, category, field, subcategory=None, num_fields=2):
        layout = QHBoxLayout()
        value_ = self.get_value(category, field, subcategory=subcategory)
        if value_ == "???":
            value_ = ["???"] * num_fields
        for value in value_:
            le = QLineEdit()
            le.setValidator(QIntValidator())
            le.setText(str(value))
            le.setEnabled(self.enabled)
            layout.addWidget(le)
        return layout

    def update_label(self, slider, label):
        label.setText(str(slider.value()))

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def check_blanks(self, parameters):
        for big_key, big_value in parameters.items():
            for key, value in big_value.items():
                if value == "???":
                    return False
        return True


def main():
    app = QApplication(sys.argv)
    project_path = os.path.join(str(Path.home()), "DLC2Action")
    settings = Project(project_path)._read_parameters(catch_blanks=False)
    window = ProjectSettings(settings, title="Hello")
    # window = TypeChoice()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
