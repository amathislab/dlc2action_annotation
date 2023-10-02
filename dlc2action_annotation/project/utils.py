from PyQt5.QtWidgets import QMessageBox


def show_error(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle("Error")
    msg.exec_()