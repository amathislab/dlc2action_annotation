from PyQt5.Qt import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout


class ProgressBar(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, window, current, loaded, length):
        super(ProgressBar, self).__init__()
        self.bar = ProgressBarWidget(window, current, loaded, length)
        self.label = CountLabel(current, length)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.bar)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.bar.clicked.connect(self.on_click)

    def on_click(self, cur):
        self.clicked.emit(cur)


class CountLabel(QLabel):
    def __init__(self, current, length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = str(length)
        self.num_symbols = len(self.length)
        self.current = current
        width = self.fontMetrics().width(f"{length}/{length}")
        self.setFixedWidth(width)
        self.setAlignment(Qt.AlignRight)

    def paintEvent(self, e):
        cur = self.current()
        self.setText(f"{str(cur).rjust(self.num_symbols)}/{self.length}")
        super().paintEvent(e)


class ProgressBarWidget(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, window, current, loaded, length):
        super(ProgressBarWidget, self).__init__()
        self.window = window
        self.len = length
        self.h = 10
        self.current_func = current
        self.get_loaded = loaded
        self.loaded = self.get_loaded()
        self.setFixedHeight(self.h)

    def paintEvent(self, e):
        self.step = (self.width() - 1) / self.len
        qp = QPainter()
        qp.begin(self)
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(168, 216, 239))
        self.loaded = self.get_loaded()
        qp.drawRect(
            self.loaded[0] * self.step,
            0,
            (self.loaded[1] - self.loaded[0]) * self.step,
            self.h,
        )
        qp.setPen(Qt.gray)
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(0, 0, self.width() - 1, self.h)
        cur = self.current_func() + 0.5
        qp.setPen(QPen(Qt.darkGray, max(2, self.step * 3)))
        qp.drawLine(cur * self.step, 0, cur * self.step, self.h)
        qp.end()

    def mousePressEvent(self, event):
        cur = int(event.x() / self.step)
        self.clicked.emit(cur)
