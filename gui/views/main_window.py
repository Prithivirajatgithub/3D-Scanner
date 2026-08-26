# gui/views/main_window.py
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QToolBar, QComboBox, QPushButton,
    QHBoxLayout, QSplitter, QStatusBar, QMessageBox, QSizePolicy, QMenu
)
from gui.styles.theme import DARK_THEME_QSS, LIGHT_THEME_QSS
from gui.views.viewports import Live2DViewWidget, Viewport3DWidget
from gui.views.control_panel import ControlPanelWidget


class MainWindow(QMainWindow):
    mode_changed = Signal(str)
    scan_triggered = Signal(bool)
    export_requested = Signal()
    open_mesh_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handheld 3D Scanner Studio - Multi-Engine Edition")
        self.resize(1360, 780)
        self.is_dark_theme = True
        self.setStyleSheet(DARK_THEME_QSS)

        self._build_toolbar()
        self._build_central_layout()
        self._build_statusbar()

    def _build_toolbar(self):
        self.toolbar = QToolBar("Main Controls")
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # 1. 3-Dot (Kebab) Settings Menu Button
        self.menu_btn = QPushButton("⋮")
        self.menu_btn.setToolTip("Options & Themes")
        self.menu_btn.setFixedSize(30, 30)
        self.menu_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 0px;
                border-radius: 4px;
            }
        """)

        self.options_menu = QMenu(self)
        self.theme_action = QAction("☀️ Switch to Light Mode", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        self.options_menu.addAction(self.theme_action)
        self.menu_btn.setMenu(self.options_menu)
        self.toolbar.addWidget(self.menu_btn)

        # 2. Mode Selector Dropdown
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "1. Multi-Engine Post-Reconstruction",
            "2. Live Real-Time Scanning"
        ])
        self.mode_combo.currentTextChanged.connect(self.mode_changed.emit)
        self.toolbar.addWidget(self.mode_combo)

        # 3. Open 3D Model Button (Next to Save As)
        self.open_mesh_btn = QPushButton("📂 Open 3D...")
        self.open_mesh_btn.setToolTip("Open and Inspect an existing 3D Model (OBJ, STL, PLY, PCD)")
        self.open_mesh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b3a4a;
                color: #00ffcc;
                border: 1px solid #3d5266;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #35495e;
                border-color: #00ffcc;
            }
        """)
        self.open_mesh_btn.clicked.connect(self.open_mesh_requested.emit)
        self.toolbar.addWidget(self.open_mesh_btn)

        # 4. Save 3D Model Button
        self.save_btn = QPushButton("💾 Save As...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.export_requested.emit)
        self.toolbar.addWidget(self.save_btn)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        # 5. Start / Stop Scan Button
        self.scan_btn = QPushButton("START SCAN")
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setCheckable(True)
        self.scan_btn.clicked.connect(self._on_scan_toggle)
        self.toolbar.addWidget(self.scan_btn)

    def _build_central_layout(self):
        central = QWidget()
        layout = QHBoxLayout(central)

        self.control_panel = ControlPanelWidget()

        splitter = QSplitter(Qt.Horizontal)
        self.view_2d = Live2DViewWidget()
        self.view_3d = Viewport3DWidget()

        splitter.addWidget(self.view_2d)
        splitter.addWidget(self.view_3d)
        splitter.setSizes([450, 650])

        layout.addWidget(self.control_panel)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | Multi-Algorithm Workspace Initialized")

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme

        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME_QSS)
            self.theme_action.setText("☀️ Switch to Light Mode")
            self.status_bar.showMessage("Switched to Dark Mode")
        else:
            self.setStyleSheet(LIGHT_THEME_QSS)
            self.theme_action.setText("🌙 Switch to Dark Mode")
            self.status_bar.showMessage("Switched to Light Mode")

        self.view_2d.set_theme_mode(self.is_dark_theme)
        self.view_3d.set_theme_mode(self.is_dark_theme)

    def _on_scan_toggle(self, checked):
        self.scan_btn.setText("STOP SCAN" if checked else "START SCAN")
        self.scan_triggered.emit(checked)

    def show_error_popup(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def show_info_popup(self, title: str, message: str):
        QMessageBox.information(self, title, message)