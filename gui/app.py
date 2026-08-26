# gui/app.py
import os
import sys
import io
import ctypes

# 1. Prevent "'NoneType' object has no attribute 'write'" in PyInstaller --windowed mode
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# 2. Fix Windows Taskbar Icon Grouping (AppUserModelID)
if sys.platform == "win32":
    try:
        myappid = "virtuascan.3dscanner.reconstruction.1.0.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

# 3. Dynamic Base Path Resolution (Handles local Python + PyInstaller bundle)
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), relative_path)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from gui.controllers.scanner_controller import ScannerController
from gui.views.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    
    # Locate icon across local development and frozen binaries
    icon_ico = get_resource_path(os.path.join("gui", "icon.ico"))
    icon_png = get_resource_path(os.path.join("gui", "icon.png"))
    
    icon_path = icon_ico if os.path.exists(icon_ico) else icon_png
    
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    
    window = MainWindow()
    window.setWindowTitle("VirtuaScan - 3D Surface Reconstruction Studio")
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))

    controller = ScannerController(window)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()