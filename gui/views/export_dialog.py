# gui/views/export_dialog.py
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt

class Export3DDialog(QDialog):
    """Clean dialog allowing clear 3D format selection without typing extensions."""
    def __init__(self, default_dir: str, default_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export 3D Asset")
        self.setFixedWidth(520)
        self.setStyleSheet("""
            QDialog { background-color: #202025; color: #ffffff; }
            QLabel { color: #d0d0d8; font-size: 12px; }
            QLineEdit { 
                background-color: #2d2d34; color: #ffffff; 
                border: 1px solid #44444e; border-radius: 4px; padding: 6px; 
            }
            QComboBox { 
                background-color: #2d2d34; color: #00ffcc; font-weight: bold;
                border: 1px solid #44444e; border-radius: 4px; padding: 6px; 
            }
            QPushButton {
                background-color: #2b3a4a; color: #ffffff; border: 1px solid #3d5266;
                border-radius: 4px; padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #35495e; }
            QPushButton#save_confirm_btn { background-color: #00887a; }
            QPushButton#save_confirm_btn:hover { background-color: #00b386; }
        """)

        self.export_dir = default_dir
        self.base_name = os.path.splitext(default_name)[0]

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # 1. Output Format Selector
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Target 3D Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("Wavefront 3D Mesh (.obj)", ".obj")
        self.format_combo.addItem("Stereolithography CAD / 3D Print (.stl)", ".stl")
        self.format_combo.addItem("Polygon File Format (.ply)", ".ply")
        self.format_combo.addItem("Point Cloud Data (.pcd)", ".pcd")
        self.format_combo.currentIndexChanged.connect(self._update_preview)
        fmt_row.addWidget(self.format_combo)
        layout.addLayout(fmt_row)

        # 2. File Name Field
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("File Name:"))
        self.name_edit = QLineEdit(self.base_name)
        self.name_edit.textChanged.connect(self._update_preview)
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        # 3. Destination Directory
        dir_row = QHBoxLayout()
        self.dir_lbl = QLineEdit(self.export_dir)
        self.dir_lbl.setReadOnly(True)
        self.browse_btn = QPushButton("Browse Folder...")
        self.browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.dir_lbl)
        dir_row.addWidget(self.browse_btn)
        layout.addLayout(dir_row)

        # 4. Final Path Preview
        self.preview_lbl = QLabel()
        self.preview_lbl.setStyleSheet("color: #00ffcc; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.preview_lbl)

        # 5. Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn = QPushButton("Export File")
        self.save_btn.setObjectName("save_confirm_btn")
        self.save_btn.clicked.connect(self.accept)
        btn_box.addWidget(self.cancel_btn)
        btn_box.addWidget(self.save_btn)
        layout.addLayout(btn_box)

        self._update_preview()

    def _browse_dir(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder", self.export_dir)
        if new_dir:
            self.export_dir = new_dir
            self.dir_lbl.setText(self.export_dir)
            self._update_preview()

    def _update_preview(self):
        ext = self.format_combo.currentData()
        name = self.name_edit.text().strip() or "reconstructed_mesh"
        name = os.path.splitext(name)[0]
        full_path = os.path.join(self.export_dir, f"{name}{ext}")
        self.preview_lbl.setText(f"Will save to: {full_path}")

    def get_selected_target(self):
        ext = self.format_combo.currentData()
        name = self.name_edit.text().strip() or "reconstructed_mesh"
        name = os.path.splitext(name)[0]
        full_path = os.path.join(self.export_dir, f"{name}{ext}")
        return full_path, ext