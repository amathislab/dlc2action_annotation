from PyQt5.Qt import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QIcon, QKeySequence
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from collections import deque


class List(QListWidget):
    def __init__(self, window, *args, **kwargs):
        super(List, self).__init__(*args, **kwargs)
        self.window = window
        self.animal_shortcuts = [
            QKeySequence(str(i)) for i in range(min(self.window.n_animals(), 10))
        ]

    def reset_shortcuts(self, shortcuts):
        self.shortcuts = [QKeySequence(key)[0] for key in shortcuts["base"]]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.window.on_play()
        elif event.key() == Qt.Key_Right:
            self.window.next()
        elif event.key() == Qt.Key_Left:
            self.window.prev()
        elif event.key() == Qt.Key_Escape:
            self.window.on_escape()
        elif event.key() == Qt.Key_Minus:
            self.window.on_minus()
        elif event.key() == Qt.Key_Return:
            self.window.on_enter()
        elif event.key() == Qt.Key_Equal:
            self.window.on_plus()
        elif event.key() in [
            QKeySequence(key)[0] for key in self.window.active_shortcuts()
        ]:
            self.window.on_shortcut(event.text().upper())
            self.window.on_shortcut(event.text().upper())
        elif event.key() in self.animal_shortcuts:
            self.window.set_animal(int(event.text()))
        else:
            super(List, self).keyPressEvent(event)
            print(f"list didn't recognise key {event.text()}")


class AnimalList(List):
    def __init__(self, current, visuals, *args, **kwargs):
        super(AnimalList, self).__init__(*args, **kwargs)
        self.update_list(current, visuals)

    def update_list(self, current, visuals):
        self.clear()
        for i, (animal, color) in enumerate(visuals):
            col = QColor(*[x * 255 for x in color])
            pixmap = QPixmap(100, 100)
            pixmap.fill(col)
            item = QListWidgetItem(f"{animal} ({i})")
            item.setIcon(QIcon(pixmap))
            self.addItem(item)
            if animal == current:
                item.setSelected(True)
            else:
                item.setSelected(False)
        self.reset_shortcuts(self.window.shortCut)


class CatList(List):
    def __init__(self, key, *args, **kwargs):
        super(CatList, self).__init__(*args, **kwargs)
        self.key = key
        self.update_list()

    def set_key(self, key):
        self.key = key
        self.update_list()
        self.window.set_cat(self.item(0))

    def update_list(self):
        cat_key = self.key
        self.clear()
        inv = self.window.shortCutInv(key=cat_key)
        for cat in self.window.catDict[cat_key]:
            if self.window.catDict[cat_key][cat] not in self.window.invisible_actions:
                col = QColor(*self.window.bar.get_color(self.window.catDict[cat_key][cat]))
                pixmap = QPixmap(100, 100)
                pixmap.fill(col)
                try:
                    sc = inv[cat]
                except:
                    sc = "no shortcut"
                item = QListWidgetItem(f'{self.window.catDict["base"][cat]} ({sc})')
                item.setIcon(QIcon(pixmap))
                self.addItem(item)
        if self.count() > 0:
            self.item(0).setSelected(True)
        self.reset_shortcuts(self.window.shortCut)


class SegmentationList(List):
    itemsChecked = pyqtSignal(list)

    def __init__(self, cats, *args, **kwargs):
        super(SegmentationList, self).__init__(*args, **kwargs)
        with open("colors.txt") as f:
            colors = [
                list(map(lambda x: float(x), line.split())) for line in f.readlines()
            ][::-1]

        for i, cat in enumerate(cats):
            if type(cat) is int:
                cat = f"category {cat}"
            item = QListWidgetItem(cat)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            col = QColor.fromRgb(*colors[i], 127)
            pixmap = QPixmap(100, 100)
            pixmap.fill(col)
            item.setIcon(QIcon(pixmap))
            self.addItem(item)
        self.update_list()
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.itemChanged.connect(self.check_items)

    def update_list(self):
        self.reset_shortcuts(self.window.shortCut)

    def check_items(self):
        checked = []
        for i in range(self.count()):
            item = self.item(i)
            if item.checkState():
                checked.append(i)
        self.itemsChecked.emit(checked)
