from dlc2action.project import Project
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


DATA_PATH = "/Users/liza/OFT/Output_DLC"
ANNOTATION_PATH = "/Users/liza/OFT/Labels"
PROJECTS_PATH = '/Users/liza/DLC2Action'
DEVICE = "cpu"

project = Project(
    "oft",
    data_type="dlc_track",
    annotation_type="csv",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH
)

project.update_parameters(
    {
        "data": {
            "canvas_shape": [928, 576],
            "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
            "annotation_suffix": ".csv",
            "behaviors": ["Grooming", "Supported", "Unsupported"],
            "ignored_bodyparts": {"tl", "tr", "br", "bl", "centre"},
            "likelihood_threshold": 0.8,
            "filter_background": False,
            "filter_annotated": False,
            "fps": 25,
            "clip_frames": 0,
            "normalize": True,
        },
        "general": {
            "model_name": "mlp",
            "exclusive": True,
            "only_load_annotated": True,
            "metric_functions": {'accuracy', 'f1'},
            "dim": 2,
            "len_segment": 512,
            "overlap": 0,
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "none"},
        },
        "training": {
            "num_epochs": 100,
            "device": DEVICE,
            "test_frac": 0.7,
            "lr": 1e-4,
            "batch_size": 32,
            "ssl_weights": {"pairwise": 0.01, "contrastive_regression": 1},
            "augment_train": 1,
            "skip_normalization_keys": ["speed_direction", "coord_diff"],
            "to_ram": False
        },
        "losses": {
            "ms_tcn": {
                "weights": 'dataset_inverse_weights',
                "gamma": 2.5,
                "alpha": 0.001,
                # "focal": False
            }
        },
        "augmentations": {
            "augmentations": {"add_noise", "mirror"},
            "mirror_dim": {0, 1},
            "noise_std": 0.001,
            "canvas_shape": [928, 576]
        },
        "features": {
            "egocentric": True,
            "distance_pairs": None,
            "keys": {"intra_distance", "angle_speeds", "areas", "coord_diff",
                     "acc_joints", "center", "speed_direction", "speed_value", "zone_bools"},
            "angle_pairs": [["tailbase", "tailcentre", "tailcentre", "tailtip"],
                         ["hipr", "tailbase", "tailbase", "hipl"],
                         ["tailbase", "bodycentre", "bodycentre", "neck"],
                         ["bcr", "bodycentre", "bodycentre", "bcl"],
                         ["bodycentre", "neck", "neck", "headcentre"],
                         ["tailbase", "bodycentre", "neck", "headcentre"]
                         ],
            "area_vertices": [["tailbase","hipr","hipl"], ["hipr","hipl","bcl","bcr"],
                           ["bcr","earr","earl","bcl"], ["earr","nose","earl"]],
            "neighboring_frames": 0,
            "zone_vertices": {"arena": ["tl", "tr", "br", "bl"]},
            "zone_bools": [["arena", "nose"], ["arena", "headcentre"]]
        },
    }
)

project.remove_saved_features()
