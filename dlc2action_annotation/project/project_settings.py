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



class TypeChoice(QWidget):
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

    def make_data_combo(self):
        combo = QComboBox()
        options = Project.data_types()
        for option in options:
            combo.addItem(option)
        combo.currentIndexChanged.connect(self.data_changed)
        return combo
    
    def data_changed(self, index=None):
        self.data_type = self.data_combo.currentText()
        self.data_label.setText(input_stores[self.data_type].__doc__)

    def make_annotation_combo(self):
        combo = QComboBox()
        options = Project.annotation_types()
        for option in options:
            combo.addItem(option)
        combo.currentIndexChanged.connect(self.annotation_changed)
        return combo
    
    def annotation_changed(self, index=None):
        self.annotation_type = self.annotation_combo.currentText()
        self.annotation_label.setText(annotation_stores[self.annotation_type].__doc__)

    def make_button(self):
        button = QPushButton("Create Project")
        button.clicked.connect(self.create_project)
        return button
    
    def accept(self):
        print("Accepted")

    def reject(self):
        print("Rejected")

    def sizeHint(self):
        return QSize(800, 500)
    

class ProjectSettings(QWidget):
    def __init__(self, project_path="/Users/liza/DLC2Action/test"):
        super().__init__()
        self.settings = Project(project_path)._read_parameters(catch_blanks=False)
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
        self.setLayout(self.layout)

    def accept(self):
        print("Accepted")
    
    def reject(self):
        print("Rejected")

    def create_general_tab(self):
        self.general_tab = QWidget()
        self.tabs.addTab(self.general_tab, "General")
        self.general_layout = QFormLayout()
        self.general_tab.setLayout(self.general_layout)

    def create_training_tab(self):
        self.training_tab = QWidget()
        self.tabs.addTab(self.training_tab, "Training")
        self.training_layout = QFormLayout()
        self.training_tab.setLayout(self.training_layout)

    def create_augmentations_tab(self):
        self.augmentation_tab = QWidget()
        self.tabs.addTab(self.augmentation_tab, "Augmentations")
        self.augmentation_layout = QFormLayout()
        self.augmentation_tab.setLayout(self.augmentation_layout)

    def create_features_tab(self):
        self.features_tab = QWidget()
        self.tabs.addTab(self.features_tab, "Features")
        self.features_layout = QFormLayout()
        self.features_tab.setLayout(self.features_layout)
    
    def create_data_tab(self):
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data")
        self.data_layout = QFormLayout()
        self.data_tab.setLayout(self.data_layout)
    
    def set_general_tab(self):
        self.clearLayout(self.general_layout)
        self.set_general_tab_data()
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
        self.training_layout.addRow("Train on multiple devices in parallel: ", self.parallel)
        self.training_layout.addRow("Validate every n epochs: ", self.validation_interval)
        self.training_layout.addRow("Save model every n epochs: ", self.model_save_epochs)
        self.training_layout.addRow("Number of epochs: ", self.num_epochs)
        self.training_layout.addRow("Batch size: ", self.batch_size)
        self.training_layout.addRow("Normalize data: ", self.normalize)
        self.training_layout.addRow("Temporal subsampling fraction: ", self.temporal_subsampling_size)
        self.training_layout.addRow("Partition method: ", self.partition_method)
        self.training_layout.addRow("Validation fraction: ", self.val_frac)
        self.training_layout.addRow("Test fraction: ", self.test_frac)

    def set_augmentations_tab(self):
        self.clearLayout(self.augmentation_layout)
        self.set_augmentations_tab_data()
        self.augmentation_layout.addRow("Augmentations: ", self.augmentations)
        self.augmentation_layout.addRow("Rotation limit: ", self.rotation_limit)
        self.augmentation_layout.addRow("Mirror dimensions: ", self.mirror_dim)
        self.augmentation_layout.addRow("Noise standard deviation: ", self.noise_std)
        self.augmentation_layout.addRow("Zoom limits: ", self.zoom_limits)

    def set_features_tab(self):
        self.clearLayout(self.features_layout)
        self.set_features_tab_data()
        self.features_layout.addRow("Features: ", self.keys)
        self.features_layout.addRow("Distance pairs: ", self.distance_pairs)

    def set_general_tab_data(self):
        self.exclusive = self.set_toggle("general", "exclusive")
        self.only_annotated = self.set_toggle("general", "only_load_annotated")
        self.ignored_agents = self.set_multiple_input("general", "ignored_clips")
        self.num_cpus = self.set_le("general", "num_cpus", set_int=True)
        self.overlap = self.set_slider("general", "overlap", 0, 1, percent=True)
        self.interactive = self.set_toggle("general", "interactive")

    def set_training_tab_data(self):
        self.lr = self.set_le("training", "lr", set_int=False, set_float=True)
        self.weight_decay = self.set_le("training", "weight_decay", set_int=False, set_float=True)
        self.device = self.set_le("training", "device", set_int=False, set_float=False)
        self.validation_interval = self.set_le("training", "validation_interval", set_int=True, set_float=False)
        self.num_epochs = self.set_le("training", "num_epochs", set_int=True, set_float=False)
        self.batch_size = self.set_le("training", "batch_size", set_int=True, set_float=False)
        self.model_save_epochs = self.set_le("training", "model_save_epochs", set_int=True, set_float=False)
        self.normalize = self.set_toggle("training", "normalize")
        self.parallel = self.set_toggle("training", "parallel")
        self.temporal_subsampling_size = self.set_slider("training", "temporal_subsampling_size", 0, 1, percent=True)
        self.partition_method = self.set_combo("training", "partition_method", ["random", "file", "random:equalize:segments", "random:equalize:videos", "folders", "time", "time:strict"])
        self.val_frac = self.set_slider("training", "val_frac", 0, 1, percent=True)
        self.test_frac = self.set_slider("training", "test_frac", 0, 1, percent=True)
        self.split_path = self.set_file("training", "split_path", dir=False)
    
    def set_augmentations_tab_data(self):
        self.augmentations = self.set_options("augmentations", "augmentations", ['rotate', 'real_lens', 'add_noise', 'shift', 'zoom', 'mirror', 'switch'])
        self.rotation_limit = self.set_limits("augmentations", "rotation_limits", -180, 180, factor=3.14/180)
        self.mirror_dim = self.set_options("augmentations", "mirror_dim", [0, 1, 2])
        self.noise_std = self.set_le("augmentations", "noise_std", set_int=False, set_float=True)
        self.zoom_limits = self.set_limits("augmentations", "zoom_limits", 0, 10)

    def set_features_tab_data(self):
        self.keys = self.set_options("features", "keys", ["coords", "coord_diff", "center", "intra_distance", "inter_distance", "speed_direction", "speed_value", "acc_joints", "likelihood"])
        self.distance_pairs = self.set_multiple_input("features", "distance_pairs", type="double")
        # not including the rest of the features because writing the input fields is a pain

    def set_data_tab_data(self):
        input_functions = {
            "calms21": self.set_calms21_input,
            "clip": self.set_clip_input,
            "dlc_track": self.set_dlc_track_input,
            "dlc_tracklet": self.set_dlc_tracklet_input,
            "features": self.set_features_input,
            "np_3d": self.set_np_3d_input,
            "pku_mmd": self.set_pku_mmd_input,
            "simba": self.set_simba_input,
        }
    
    def set_calms21_input(self):
        self.treba_files = self.set_toggle("data", "treba_files")
        self.task_n = self.set_combo("data", "task_n", [1, 2])
    
    def set_clip_input(self):
        self.tracking_suffix = self.set_le("data", "tracking_suffix", set_int=False, set_float=False)
        self.tracking_path = self.set_file("data", "tracking_path", dir=True)
        self.frame_limit = self.set_le("data", "frame_limit", set_int=True, set_float=False)
        self.default_agent_name = self.set_le("data", "default_agent_name", set_int=False, set_float=False)
    
    def set_dlc_track_input(self):
        self.data_suffix = self.set_le("data", "data_suffix", set_int=False, set_float=False)
        self.data_prefix = self.set_le("data", "data_prefix", set_int=False, set_float=False)
        self.feature_suffix = self.set_le("data", "feature_suffix", set_int=False, set_float=False)
        self.convert_int_indices = self.set_toggle("data", "convert_int_indices")
        self.canvas_shape = self.set_dimensions("data", "canvas_shape")
        self.ignored_bodyparts = self.set_multiple_input("data", "ignored_bodyparts")
        self.default_agent_name = self.set_le("data", "default_agent_name", set_int=False, set_float=False)
        self.likelihood_threshold = self.set_le("data", "likelihood_threshold", set_int=False, set_float=True)

    def set_dlc_tracklet_input(self):
        self.data_suffix = self.set_le("data", "data_suffix", set_int=False, set_float=False)
        self.data_prefix = self.set_le("data", "data_prefix", set_int=False, set_float=False)
        self.feature_suffix = self.set_le("data", "feature_suffix", set_int=False, set_float=False)
        self.convert_int_indices = self.set_toggle("data", "convert_int_indices")
        self.canvas_shape = self.set_dimensions("data", "canvas_shape")
        self.ignored_bodyparts = self.set_multiple_input("data", "ignored_bodyparts")
        self.default_agent_name = self.set_le("data", "default_agent_name", set_int=False, set_float=False)
        self.likelihood_threshold = self.set_le("data", "likelihood_threshold", set_int=False, set_float=True)
        self.frame_limit = self.set_le("data", "frame_limit", set_int=True, set_float=False)
    
    def set_features_input(self):
        self.feature_suffix = self.set_le("data", "feature_suffix", set_int=False, set_float=False)
        self.default_agent_name = self.set_le("data", "default_agent_name", set_int=False, set_float=False)
        self.frame_limit = self.set_le("data", "frame_limit", set_int=True, set_float=False)
    
    def set_np_3d_input(self):
        self.data_suffix = self.set_le("data", "data_suffix", set_int=False, set_float=False)
        self.feature_suffix = self.set_le("data", "feature_suffix", set_int=False, set_float=False)
        self.canvas_shape = self.set_dimensions("data", "canvas_shape")
        self.ignored_bodyparts = self.set_multiple_input("data", "ignored_bodyparts")
        self.default_agent_name = self.set_le("data", "default_agent_name", set_int=False, set_float=False)
        self.frame_limit = self.set_le("data", "frame_limit", set_int=True, set_float=False)
    
    def set_pku_mmd_input(self):
        pass

    def set_simba_input(self):
        self.data_suffix = self.set_le("data", "data_suffix", set_int=False, set_float=False)
        self.data_prefix = self.set_le("data", "data_prefix", set_int=False, set_float=False)
        self.feature_suffix = self.set_le("data", "feature_suffix", set_int=False, set_float=False)
        self.canvas_shape = self.set_dimensions("data", "canvas_shape")
        self.ignored_bodyparts = self.set_multiple_input("data", "ignored_bodyparts")
        self.likelihood_threshold = self.set_le("data", "likelihood_threshold", set_int=False, set_float=True)
        self.centered = self.set_toggle("data", "centered")
        self.use_features = self.set_toggle("data", "use_features")
        
    def collect_general(self):
        self.settings["general"]["exclusive"] = self.exclusive.isChecked()
        self.settings["general"]["only_load_annotated"] = self.only_annotated.isChecked()
        self.settings["general"]["ignored_clips"] = self.ignored_agents.get_data()
        self.settings["general"]["num_cpus"] = int(self.num_cpus.text())
        self.settings["general"]["overlap"] = self.overlap.value() / 100
        self.settings["general"]["interactive"] = self.interactive.isChecked()
    
    def collect_training(self):
        self.settings["training"]["lr"] = float(self.lr.text())
        self.settings["training"]["weight_decay"] = float(self.weight_decay.text())
        self.settings["training"]["device"] = self.device.text()
        self.settings["training"]["validation_interval"] = int(self.validation_interval.text())
        self.settings["training"]["num_epochs"] = int(self.num_epochs.text())
        self.settings["training"]["batch_size"] = int(self.batch_size.text())
        self.settings["training"]["model_save_epochs"] = int(self.model_save_epochs.text())
        self.settings["training"]["normalize"] = self.normalize.isChecked()
        self.settings["training"]["parallel"] = self.parallel.isChecked()
        self.settings["training"]["temporal_subsampling_size"] = self.temporal_subsampling_size.value() / 100
        self.settings["training"]["partition_method"] = self.partition_method.currentText()
        self.settings["training"]["val_frac"] = self.val_frac.value() / 100
        self.settings["training"]["test_frac"] = self.test_frac.value() / 100
        self.settings["training"]["split_path"] = self.split_path[0].text()

    def collect_augmentations(self):
        self.settings["augmentations"]["augmentations"] = set([x.text() for x in self.augmentations.findChildren(QCheckBox) if x.isChecked()])
        self.settings["augmentations"]["rotation_limits"] = [float(x.text()) for x in self.rotation_limit.findChildren(QLineEdit)]
        self.settings["augmentations"]["mirror_dim"] = set([int(x.text()) for x in self.mirror_dim.findChildren(QCheckBox) if x.isChecked()])
        self.settings["augmentations"]["noise_std"] = float(self.noise_std.text())
        self.settings["augmentations"]["zoom_limits"] = [float(x.text()) for x in self.zoom_limits.findChildren(QLineEdit)]

    def set_toggle(self, category, field):
        toggle = QCheckBox()
        value = self.settings[category][field]
        if value != "???":
            toggle.setChecked(self.settings[category][field])
        else:
            toggle.tristate = True
            toggle.setCheckState(Qt.PartiallyChecked)
        return toggle
    
    def set_multiple_input(self, category, field, type="single"):
        if self.settings[category][field] is None:
            if type == "category":
                x = {}
            else:
                x = []
        else:
            x = self.settings[category][field]
        if type == "double":
            widget = MultipleDoubleInputWidget(x)
        elif type == "category":
            widget = CategoryInputWidget(x)
        else:
            widget = MultipleInputWidget(x)
        return widget
    
    def set_le(self, category, field, set_int=True, set_float=False):
        le = QLineEdit()
        if set_int:
            le.setValidator(QIntValidator())
        elif set_float:
            le.setValidator(QDoubleValidator())
        le.setText(str(self.settings[category][field]))
        return le
    
    def set_slider(self, category, field, minimum, maximum, percent=False):
        value = self.settings[category][field]
        if percent:
            minimum *= 100
            maximum *= 100
            value *= 100
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        return slider
    
    def set_combo(self, category, field, options):
        combo = QComboBox()
        for o in options:
            combo.addItem(o)
        combo.setCurrentIndex(options.index(self.settings[category][field]))
        return combo
    
    def set_file(self, category, field, filter=None, dir=False):
        layout = QHBoxLayout()
        file = self.settings[category][field]
        file = file if file is not None else "None"
        button = QPushButton("Find")
        label = QLabel(os.path.basename(file))
        if dir:
            button.clicked.connect(lambda: self.get_dir(label, field, filter))
        else:
            button.clicked.connect(lambda: self.get_file(label, field, filter))
        layout.addWidget(label)
        layout.addWidget(button)
        return layout
    
    def set_options(self, category, field, options):
        layout = QFormLayout()
        for option in options:
            toggle = QCheckBox(str(option))
            toggle.setChecked(option in self.settings[category][field])
            layout.addRow(toggle)
        return layout
    
    def set_limits(self, category, field, minimum, maximum, middle=0, factor=1):
        layout = QHBoxLayout()
        min_value, max_value = self.settings[category][field]
        min_value /= factor
        max_value /= factor
        layout.addWidget(QLabel("Min: "))
        le = QLineEdit()
        le.setValidator(QDoubleValidator())
        le.setText(str(min_value))
        layout.addWidget(le)
        layout.addWidget(QLabel("Max: "))
        le = QLineEdit()
        le.setValidator(QDoubleValidator())
        le.setText(str(max_value))
        layout.addWidget(le)
        return layout
    
    def set_dimensions(self, category, field):
        layout = QHBoxLayout()
        for value in self.settings[category][field]:
            le = QLineEdit()
            le.setValidator(QIntValidator())
            le.setText(str(value))
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

    


def main():
    app = QApplication(sys.argv)

    window = ProjectSettings()
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()