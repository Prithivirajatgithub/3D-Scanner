# gui/views/control_panel.py
import torch
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QPushButton, 
    QLabel, QProgressBar, QHBoxLayout, QDoubleSpinBox
)
from PySide6.QtCore import Signal


class ControlPanelWidget(QWidget):
    session_selected = Signal(str)
    reconstruct_requested = Signal(str, str, dict)
    open_folder_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # 1. Dataset Session Manager
        sess_group = QGroupBox("1. Dataset Session")
        sess_layout = QVBoxLayout(sess_group)
        
        self.session_combo = QComboBox()
        self.session_combo.currentTextChanged.connect(self._on_session_changed)
        sess_layout.addWidget(self.session_combo)

        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Open Folder...")
        self.browse_btn.clicked.connect(self.open_folder_requested.emit)
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.browse_btn)
        btn_row.addWidget(self.refresh_btn)
        sess_layout.addLayout(btn_row)

        self.session_info_lbl = QLabel("Frames: 0 | Intrinsics: None")
        self.session_info_lbl.setStyleSheet("color: #a0a0a8; font-size: 11px;")
        sess_layout.addWidget(self.session_info_lbl)
        layout.addWidget(sess_group)

        # 2. Reconstruction Engine Box
        algo_group = QGroupBox("2. Reconstruction Engine")
        algo_layout = QVBoxLayout(algo_group)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "Open3D Scalable TSDF (Real-Time)",
            "BundleSDF Neural NeRF (High Quality)",
            "Poisson Surface Reconstruction"
        ])
        self.algo_combo.currentTextChanged.connect(self._update_hardware_target)
        algo_layout.addWidget(self.algo_combo)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Voxel Size (m):"))
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(4)
        self.voxel_spin.setRange(0.0010, 0.0500)
        self.voxel_spin.setSingleStep(0.0005)
        self.voxel_spin.setValue(0.0040)
        param_row.addWidget(self.voxel_spin)
        algo_layout.addLayout(param_row)

        self.run_recon_btn = QPushButton("RUN RECONSTRUCTION")
        self.run_recon_btn.setStyleSheet("background-color: #00887a; color: white; font-weight: bold; padding: 8px;")
        self.run_recon_btn.clicked.connect(self._on_run_clicked)
        algo_layout.addWidget(self.run_recon_btn)
        layout.addWidget(algo_group)

        # 3. Task Status Box
        prog_group = QGroupBox("3. Task Status")
        prog_layout = QVBoxLayout(prog_group)
        self.status_lbl = QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("color: #00ffcc; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.status_lbl)
        prog_layout.addWidget(self.progress_bar)
        layout.addWidget(prog_group)

        # 4. Hardware Acceleration Monitor Box
        hw_group = QGroupBox("4. Active Processing Device")
        hw_layout = QVBoxLayout(hw_group)
        hw_layout.setSpacing(6)

        # Active Device Indicator Badge
        self.dev_badge = QLabel("⚡ READY")
        self.dev_badge.setStyleSheet("""
            background-color: #23232a;
            color: #00ffcc;
            font-weight: bold;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #383844;
        """)
        hw_layout.addWidget(self.dev_badge)

        # Device Specifications Label
        self.dev_detail_lbl = QLabel("Detecting hardware...")
        self.dev_detail_lbl.setStyleSheet("color: #a0a0a8; font-size: 11px;")
        self.dev_detail_lbl.setWordWrap(True)
        hw_layout.addWidget(self.dev_detail_lbl)

        layout.addWidget(hw_group)
        layout.addStretch()

        # Initialize hardware detection
        self._init_hardware_detection()
        self._update_hardware_target()

    def _init_hardware_detection(self):
        """Probes system for CUDA GPU availability and memory."""
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.gpu_info = f"GPU: {dev_name} ({vram_gb:.1f} GB VRAM)"
        else:
            self.gpu_info = "GPU: None (CUDA unavailable)"
        self.cpu_info = "CPU: OpenMP Multithreaded Engine"

    def _update_hardware_target(self):
        """Updates indicator to show default readiness based on selected engine."""
        algo = self.algo_combo.currentText()
        if self.has_cuda:
            self.dev_badge.setText("🟢 ACTIVE: NVIDIA CUDA GPU")
            self.dev_badge.setStyleSheet("background-color: #1a382b; color: #00ff88; font-weight: bold; padding: 4px; border: 1px solid #00aa55; border-radius: 4px;")
            if "BundleSDF" in algo:
                self.dev_detail_lbl.setText(f"{self.gpu_info}\nUsing PyTorch + TensorRT Neural Solver")
            else:
                self.dev_detail_lbl.setText(f"{self.gpu_info}\nUsing Open3D Tensor GPU TSDF Engine")
        else:
            self.dev_badge.setText("🔵 ACTIVE: CPU FALLBACK")
            self.dev_badge.setStyleSheet("background-color: #1b2a3a; color: #33bbff; font-weight: bold; padding: 4px; border: 1px solid #2277aa; border-radius: 4px;")
            self.dev_detail_lbl.setText(f"{self.cpu_info}\nCUDA Unavailable, using multithreaded CPU")

    def set_active_device_state(self, active_device: str):
        """Updates indicator in real time based on active pipeline stage."""
        if active_device == "GPU":
            self.dev_badge.setText("⚡ SOLVING ON GPU (CUDA)...")
            self.dev_badge.setStyleSheet("background-color: #2b441a; color: #77ff33; font-weight: bold; padding: 4px; border: 1px solid #55aa22; border-radius: 4px;")
            self.dev_detail_lbl.setText(f"{self.gpu_info}\nAccelerated Tensor Pipeline Active")
        elif active_device == "CPU":
            self.dev_badge.setText("⚙️ TRACKING ON CPU (OpenMP)...")
            self.dev_badge.setStyleSheet("background-color: #1b3044; color: #55ccff; font-weight: bold; padding: 4px; border: 1px solid #3388cc; border-radius: 4px;")
            self.dev_detail_lbl.setText(f"{self.cpu_info}\nComputing Multithreaded Odometry")
        else:
            self._update_hardware_target()
    
    def set_sessions(self, sessions):
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for s in sessions:
            self.session_combo.addItem(s.name, s.path)
        self.session_combo.blockSignals(False)
        if sessions:
            self._on_session_changed(self.session_combo.currentText())

    def _on_session_changed(self, text):
        path = self.session_combo.currentData()
        if path:
            self.session_selected.emit(path)

    def _on_run_clicked(self):
        path = self.session_combo.currentData()
        algo = self.algo_combo.currentText()
        params = {"voxel_size": self.voxel_spin.value()}
        if path:
            self.reconstruct_requested.emit(path, algo, params)