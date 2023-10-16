from PyQt5.QtWidgets import QMessageBox


def show_error(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(message)
    msg.setWindowTitle("Error")
    msg.exec_()

def show_warning(message, details=None):
    msg = QMessageBox()
    msg.setText(message)
    msg.setIcon(QMessageBox.Warning)
    if details is not None:
        msg.setInformativeText(details)
    msg.setWindowTitle("Warning")
    msg.exec_()