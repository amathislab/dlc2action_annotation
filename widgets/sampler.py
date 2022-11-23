#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
import numpy as np


class Sampler:
    def __init__(self, classes):
        self.results = {
            "good/bad": {c: [] for c in classes},
            "edit %": {c: [] for c in classes},
        }
        self.method = "good/bad"
        self.behavior = None
        self.threshold = 0.5

    def assessment(self):
        if self.behavior is None:
            return False
        else:
            return True

    def classes(self):
        return list(self.results["good/bad"].keys())

    def get_behavior(self):
        return self.behavior

    def get_method(self):
        return self.method

    def get_threshold(self):
        return self.threshold

    def set_method(self, method):
        self.method = method

    def set_threshold(self, value):
        self.threshold = value

    def update_good(self, value):
        self.results[self.method][self.behavior].append(value)

    def update_edit(self, old_times, new_times_list):
        start, end = old_times
        interval = np.zeros(end - start)
        for s, e in new_times_list:
            interval[max(s - start, 0) : min(e - start, end - start)] = 1
        self.results[self.method][self.behavior].append(
            np.sum(interval) / (end - start)
        )
        interval = np.array([1] + list(interval) + [1])
        diffs = np.diff(interval)
        starts = list(np.argwhere(diffs == -1) + start)
        ends = list(np.argwhere(diffs == 1) + start)
        return starts, ends

    def undo(self):
        if len(self.results[self.method][self.behavior]) > 0:
            self.results[self.method][self.behavior].pop(-1)
            return True
        else:
            return False

    def start_sampling(self, behavior, method=None):
        if method is not None:
            self.method = method
        self.results[self.method][behavior] = []
        self.behavior = behavior

    def compute(self, method=None, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        if method is not None:
            self.method = method
        self.behavior = None
        values = {}
        if self.method == "good/bad":
            for key, value in self.results["good/bad"].items():
                values[key] = f"{sum(value)}/{len(value)}"
        else:
            for key, value in self.results["edit %"].items():
                values[key] = f"{np.sum(np.array(value) > self.threshold)}/{len(value)}"
        return values
