# gui/app.py
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from gui.controllers.scanner_controller import ScannerController
from gui.views.main_window import MainWindow
from PySide6.QtWidgets import QApplication


def main():
  app = QApplication(sys.argv)
  window = MainWindow()
  controller = ScannerController(window)

  window.show()
  sys.exit(app.exec())


if __name__ == '__main__':
  main()