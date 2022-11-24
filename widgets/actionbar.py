#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in https://github.com/AlexEMG/DLC2action/LICENSE.AGPL.
#
from PyQt5.QtCore import QSize
from PyQt5.Qt import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFontMetrics
import numpy as np
import math
from PyQt5.QtWidgets import QWidget
from collections import defaultdict
from copy import copy
from utils import get_color


class ActionRect:
    def __init__(self, cat, times, row=None, amb=False):
        self.cat = cat
        self.amb = amb
        if len(times) == 2:
            self.times = times
        else:
            self.times = [times[0], times[1]]
            self.amb = times[2]
        self.row = row
        self.moving_end = None
        self.highlight = False

    def set_end(self, end, value):
        self.times[end] = value

    def move(self, end):
        self.stop_suggestion()
        self.moving_end = end

    def stop(self):
        self.moving_end = None
        self.highlight = False

    def stop_suggestion(self):
        if self.amb == 2:
            self.amb = 0


class Bar(QWidget):
    clicked = pyqtSignal(int)

    def __init__(
        self,
        window,
        segment_len,
        al_mode,
        al_current,
        al_end,
        correct_animal,
        mode,
        fm,
        active=True,
    ):
        super().__init__()
        self.h = window.settings["actionbar_width"]
        self.window = window
        self.active = active
        self.move_rect = None
        self.minus_rect = None
        self.plus_rect = None
        self.al_mode = al_mode
        self.al_current = al_current
        self.al_end = al_end
        self.mode = mode
        self.correct_animal = correct_animal
        self.fm = fm
        self.grow_rects = []
        self.cat = 0
        self.start = 0
        self.len = segment_len
        self.setMouseTracking(True)
        self.get_labels()
        with open("colors.txt") as f:
            self.colors = [
                list(map(lambda x: float(x), line.split())) for line in f.readlines()
            ]

    def get_color(self, name):
        if isinstance(name, int):
            name = self.window.catDict["base"][name]
        return get_color(self.colors, name)

    def reset(self):
        self.move_rect = None
        self.grow_rects = []
        self.cat = 0
        self.get_labels()

    def paintEvent(self, e):
        self.spacing = 4
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        if self.al_mode:
            self.drawBrackets(qp)
        self.drawActions(qp)
        self.drawCursor(qp)
        qp.end()

    def drawBrackets(self, qp):
        qp.setPen(QPen(Qt.red, 3))
        size = self.size()
        start = self.al_current - self.start
        stop = self.al_end - self.start

        if start >= 0:
            x = start * self.step
            qp.drawLine(x, 0, x, math.floor(size.height()))
        if stop <= self.len:
            x = stop * self.step
            qp.drawLine(x, 0, x, math.floor(size.height()))
        start = max([0, start])
        stop = min([self.len, stop])
        qp.drawLine(
            start * self.step,
            math.floor(size.height()) - 1,
            stop * self.step,
            math.floor(size.height()) - 1,
        )
        qp.drawLine(start * self.step, 0, stop * self.step, 0)

        col = QColor("gray")
        col.setAlphaF(0.25)
        qp.setBrush(col)
        qp.setPen(col)
        if stop > 0 and start < self.len:
            qp.drawRect(0, 0, start * self.step, math.floor(size.height()) - 1)
            qp.drawRect(
                stop * self.step, 0, self.len * self.step, math.floor(size.height()) - 1
            )
        else:
            qp.drawRect(0, 0, (self.len + 1) * self.step, math.floor(size.height()) - 1)

    def drawLines(self, qp):
        qp.setPen(Qt.gray)
        size = self.size()
        self.step = size.width() / (self.len + 1)

        for i in range(self.len + 1):
            x = i * self.step
            qp.drawLine(x, 0, x, math.floor(size.height()))

    def drawCursor(self, qp):
        cur = self.local_cur() + 0.5
        qp.setPen(QPen(Qt.gray, self.step))
        qp.drawLine(
            cur * self.step, 0, cur * self.step, math.floor(self.size().height())
        )

    def add_rect(self, frame, cat, row, on_shortcut=False):
        if on_shortcut and not self.correct_animal:
            ok = self.window.show_question(
                message="This tracklet has ended. Annotate anyway?", default="no"
            )
        else:
            ok = True
        if ok:
            rect = ActionRect(cat=cat, times=[frame, frame + 1], row=row)
            self.rects.append(rect)
            self.rows[row].append(rect)
            rect.move(1)
            if self.mode == "N" and not on_shortcut:
                self.move_rect = rect
            else:
                self.grow_rects.append(rect)

    def create_init_rect(self, next, times, categories, row):
        if row is None:
            row = self.nrows
        rect = ActionRect(cat=categories[next], times=list(times[next][0, :]), row=row)
        self.rects.append(rect)
        self.rows[row].append(rect)
        times[next] = times[next][1:, :]
        if len(times[next]) == 0:
            times.pop(next)
            categories.pop(next)
        return (times, categories)

    def create_rects(self):
        self.rects = []
        self.rows = defaultdict(lambda: [])
        categories = list(range(len(self.times)))
        self.times = [sorted(x, key=lambda x: x[0]) for x in self.times]
        for i in categories:
            to_remove = []
            for j in range(len(self.times[i])):
                if self.times[i][j][1] < self.times[i][j][0]:
                    self.times[i][j] = [
                        self.times[i][j][1],
                        self.times[i][j][0],
                        self.times[i][j][2],
                    ]
                elif self.times[i][j][1] == self.times[i][j][0]:
                    to_remove.append(self.times[i][j])
            for element in to_remove:
                self.times[i] = [
                    x for x in self.times[i] if not np.array_equal(x, element)
                ]

        times = copy(self.times)
        for i, x in enumerate(times):
            times[i] = sorted(x, key=lambda x: x[0])
        remove_list_times = []
        remove_list_cats = []
        for i in categories:
            if len(times[i]) == 0:
                remove_list_times.append(times[i])
                remove_list_cats.append(i)
        for x in remove_list_times:
            times.remove(x)
        for x in remove_list_cats:
            categories.remove(x)
        if len(times) > 0 and type(times[0]) is not np.ndarray:
            for i in range(len(times)):
                times[i] = np.array(
                    [[int(x[0]), int(x[1]), int(x[2])] for x in times[i]]
                )
        while len(times) > 0:
            starts = np.array([x[0, 0] for x in times])
            next = np.argmin(starts)
            for row in range(self.nrows):
                if (
                    len(self.rows[row]) == 0
                    or self.rows[row][-1].times[1] <= times[next][0, 0]
                ):
                    times, categories = self.create_init_rect(
                        next, times, categories, row
                    )
                    break

    def reorder_rects(self):
        self.set_data_from_rects()
        self.rows = defaultdict(lambda: [])
        rects_sorted = self.rects.copy()
        rects_sorted.sort(key=lambda x: x.times[0])
        for rect in rects_sorted:
            for row in range(self.nrows):
                if (
                    len(self.rows[row]) == 0
                    or self.rows[row][-1].times[1] < rect.times[0]
                ):
                    self.rows[row].append(rect)
                    rect.row = row
                    break

    def drawActions(self, qp):
        self.rowh = self.size().height() // self.nrows
        if self.window.current() == 0:
            self.stop_growing()
        if len(self.grow_rects) > 0:
            for rect in self.grow_rects:
                rect.times[1] = self.window.current()
                self.check_merge(rect)
            self.reorder_rects()
        for rect in [x for x in self.rects if x.row is not None]:
            if rect.highlight:
                qp.setPen(QPen(Qt.blue, 3))
            elif rect.amb == 2:
                qp.setPen(QPen(Qt.white, 3))
            elif rect.amb == 3:
                qp.setPen(QPen(Qt.black, 1))
            else:
                qp.setPen(QPen(Qt.gray, 1))
            start, end = np.array(rect.times) - self.start
            if start > self.len - 1 or end < 0:
                continue
            if start < 0:
                start = 0
            if end > self.len:
                end = self.len
            col = QColor(*self.get_color(rect.cat))
            if rect.amb == 2:
                col.setAlphaF(0.3)
            elif rect.amb == 3:
                col.setAlphaF(0.8)
            elif rect.amb:
                col.setAlphaF(0.6)
            else:
                col.setAlphaF(1)
            qp.setBrush(col)
            qp.drawRect(
                start * self.step,
                rect.row * self.rowh + self.spacing,
                (end - start) * self.step,
                self.rowh - 2 * self.spacing,
            )
            text = self.window.catDict["base"][rect.cat]
            pixelsWide = self.fm.width(text)
            pixelsHigh = self.fm.height()
            if (
                pixelsWide < (end - start) * self.step / 2
                and pixelsHigh < self.rowh - self.spacing * 2
            ):
                qp.setPen(Qt.black)
                qp.drawText(
                    ((start + end) * self.step - pixelsWide) / 2,
                    (rect.row + 1 / 2) * self.rowh + pixelsHigh / 3,
                    text,
                )
                qp.setPen(Qt.gray)

    def mousePressEvent(self, event):
        if not self.active:
            return
        var = 3
        move_row = None
        change = False
        for row in range(self.nrows):
            if (
                event.y() > row * self.rowh + self.spacing
                and event.y() < (row + 1) * self.rowh - self.spacing
            ):
                move_row = row
        if move_row is not None:
            if self.mode in ["R", "C", "A", "As"]:
                for rect in [x for x in self.rects if x.row == move_row]:
                    start, end = rect.times
                    x = self.global_cur(int(event.x() // self.step))
                    if x >= start and x + 1 <= end:
                        change = True
                        if self.mode == "A":
                            if rect.amb != 3:
                                rect.amb = not rect.amb
                        elif self.mode == "As":
                            rect.stop_suggestion()
                            rect.cat = self.cat
                        else:
                            self.rects.remove(rect)
                            if self.mode == "C":
                                new_rect = ActionRect(
                                    cat=rect.cat, times=[start, x], amb=rect.amb
                                )
                                self.rects.append(new_rect)
                                rect.set_end(0, x + 1)
                                self.rects.append(rect)
                            self.reorder_rects()
                        self.update()
                        self.window.set_move_mode()
                        break
            elif self.mode == "N":
                change = True
                frame = self.global_cur(int(event.x() // self.step))
                self.add_rect(frame, self.cat, move_row)
                self.window.set_move_mode()
                self.reorder_rects()
            else:
                for rect in [x for x in self.rects if x.row == move_row]:
                    start, end = np.array(rect.times) - self.start
                    if np.abs(event.x() - start * self.step) <= var:
                        self.move_rect = rect
                        rect.move(0)
                    elif np.abs(event.x() - end * self.step) <= var:
                        self.move_rect = rect
                        rect.move(1)

        if not self.move_rect and not change:
            cur = self.global_cur(int(event.x() // self.step))
            self.clicked.emit(cur)

    def mouseReleaseEvent(self, event):
        if not self.active:
            return
        if self.move_rect:
            if self.move_rect.times[1] - self.move_rect.times[0] == 0:
                self.remove_rect(self.move_rect)
            self.stop_rect(self.move_rect)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.move_rect and self.active:
            ends = list(self.move_rect.times)
            ends[self.move_rect.moving_end] = self.global_cur(
                int(event.x() / self.step)
            )
            if ends[0] < 0:
                ends[0] = 0
            if ends[1] - ends[0] < 0:
                self.move_rect.times = self.move_rect.times[::-1]
                self.move_rect.move(1 - self.move_rect.moving_end)
            else:
                self.move_rect.times = ends
                self.check_merge(self.move_rect)
                self.reorder_rects()
                self.update()
        else:
            move_row = None
            message = self.window.message
            for row in range(self.nrows):
                if (
                    event.y() > row * self.rowh + self.spacing
                    and event.y() < (row + 1) * self.rowh - self.spacing
                ):
                    move_row = row
            if move_row is not None:
                x = self.global_cur(int(event.x() // self.step))
                for rect in [r for r in self.rects if r.row == move_row]:
                    start, end = rect.times
                    if x >= start and x < end:
                        message = self.window.catDict["base"][rect.cat]
            self.window.status.emit(message)

    def sizeHint(self):
        return QSize(400, 300)

    def get_labels(self):
        self.times = self.window.times[self.window.current_animal()]
        self.set_labels()
        self.create_rects()

    def set_labels(self):
        labels = np.zeros((len(self.times), self.window.video_len()))
        for cat in range(len(self.times)):
            to_remove = []
            for start, end, amb in self.times[cat]:
                start = int(start)
                end = int(end)
                if np.sum(labels[cat, start:end]) != 0:
                    to_remove.append([start, end, amb])
                if amb == 2:
                    labels[cat, start:end] = 0.1
                elif amb == 3:
                    labels[cat, start:end] = 0.7
                elif amb:
                    labels[cat, start:end] = 0.5
                else:
                    labels[cat, start:end] = 1
            for element in to_remove:
                self.times[cat] = [
                    x for x in self.times[cat] if not np.array_equal(x, element)
                ]
        # self.labels = labels
        self.nrows = int(np.max(np.sum(labels > 0, axis=0))) + 1
        # print(f'{self.labels[np.sum(labels > 0, axis=0) == self.nrows]=}')

    # def set_times_from_labels(self):
    #     times = []
    #     for i in range(self.labels.shape[0]):
    #         l = copy(self.labels[i, :])
    #         l = (l == 0.5).astype(int)
    #         list_amb = [[*x, 1] for x in
    #                     np.flatnonzero(np.diff(np.r_[0, l, 0]) != 0).reshape(-1, 2)]
    #         l = copy(self.labels[i, :])
    #         l = (l == 1).astype(int)
    #         list_sure = [[*x, 0] for x in
    #                     np.flatnonzero(np.diff(np.r_[0, l, 0]) != 0).reshape(-1, 2)]
    #         l = (l == 0.1).astype(int)
    #         list_suggested = [[*x, 0.5] for x in
    #                      np.flatnonzero(np.diff(np.r_[0, l, 0]) != 0).reshape(-1, 2)]
    #         times.append(np.array(list_amb + list_sure + list_suggested))
    #     self.times = times

    def set_data_from_rects(self, final=False):
        self.times = [[] for i in range(self.window.ncat)]
        to_remove = []
        for rect in self.rects:
            if final:
                if rect.times[1] == rect.times[0]:
                    to_remove.append(rect)
                    continue
                elif rect.times[1] < rect.times[0]:
                    rect.times = rect.times[::-1]
            t = [*rect.times, rect.amb]
            self.times[rect.cat].append(t)
        for rect in to_remove:
            self.rects.remove(rect)
        self.set_labels()

    def set_cat(self, cat_name):
        self.cat = self.window.catDictInv()[cat_name]

    def remove_rect(self, rect):
        self.rects.remove(rect)
        self.reorder_rects()

    def on_shortcut(self, sc):
        add_new = True
        key = self.window.active_list
        cat = self.window.shortCut[key][sc]
        rects = self.grow_rects.copy()
        for rect in rects:
            if rect.cat == cat:
                add_new = False
                if self.minus_rect == rect:
                    self.move_minus()
                if self.plus_rect == rect:
                    self.move_plus()
                rect.stop()
                self.grow_rects.remove(rect)
        if add_new and key not in ["base", "categories"]:
            cat_key = self.window.catDictInv("categories")[key]
            for rect in self.grow_rects:
                if rect.cat == cat_key:
                    add_new = False
                    rect.cat = cat
        if add_new:
            self.add_rect(self.window.current(), cat, None, on_shortcut=True)
            self.reorder_rects()
        self.window.set_move_mode()

    def check_merge(self, move_rect):
        rect_start, rect_end = move_rect.times
        for r in [x for x in self.rects if x.cat == move_rect.cat and x != move_rect]:
            s, e = r.times
            if s <= rect_end and e >= rect_start:
                move_rect.times = [min((rect_start, s)), max(e, rect_end)]
                self.rects.remove(r)
                self.stop_rect(move_rect)
                break

    def stop_growing(self, save=False):
        self.grow_rects = []

    def stop_rect(self, rect):
        if self.move_rect == rect:
            self.move_rect = None
        elif rect in self.grow_rects:
            self.grow_rects.remove(rect)

    def local_cur(self):
        return self.window.current() - self.start

    def global_cur(self, local):
        return local + self.start

    def set_minus_mode(self, value):
        if value and len(self.grow_rects) > 0:
            self.minus_rect = self.grow_rects[0]
            self.minus_rect.highlight = True
        elif self.minus_rect is not None:
            self.minus_rect.highlight = False
            self.minus_rect = None

    def set_plus_mode(self, value):
        if value and len(self.grow_rects) > 0:
            self.plus_rect = self.grow_rects[0]
            self.plus_rect.highlight = True
        elif self.plus_rect is not None:
            self.plus_rect.highlight = False
            self.plus_rect = None

    def move_minus(self):
        ind = self.grow_rects.index(self.minus_rect) + 1
        self.minus_rect.highlight = False
        self.minus_rect = self.grow_rects[ind % len(self.grow_rects)]
        self.minus_rect.highlight = True

    def move_plus(self):
        ind = self.grow_rects.index(self.plus_rect) + 1
        self.plus_rect.highlight = False
        self.plus_rect = self.grow_rects[ind % len(self.grow_rects)]
        self.plus_rect.highlight = True

    def stop_minus(self):
        self.minus_rect.stop()
        self.grow_rects.remove(self.minus_rect)
        self.minus_rect = None

    def stop_plus(self):
        if self.plus_rect.amb not in [2, 3]:
            self.plus_rect.amb = not self.plus_rect.amb
        self.plus_rect.highlight = False
        self.plus_rect = None

    def set_al_point(self, cur, end):
        self.stop_growing()
        self.al_current = cur
        self.al_end = end

    def get_new_times(self, behavior_ind):
        self.set_data_from_rects()
        new_times = []
        for start, end, amb in self.times[behavior_ind]:
            if start < self.al_end and end > self.al_current:
                new_times.append([start, end])
        return new_times
