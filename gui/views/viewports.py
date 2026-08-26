# gui/views/viewports.py
import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QVector3D, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QGroupBox, QLabel, QVBoxLayout, QPushButton, QWidget, QHBoxLayout, QSizePolicy
)


class CloseViewportButton(QPushButton):
    """Custom vector-rendered circular close button with crisp anti-aliased cross."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Close 3D Model")
        self._hover = False

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._hover:
            bg_color = QColor(220, 53, 69, 230)
            border_color = QColor(255, 100, 110, 255)
            x_color = QColor(255, 255, 255)
        else:
            bg_color = QColor(32, 33, 38, 200)
            border_color = QColor(70, 72, 82, 220)
            x_color = QColor(240, 70, 80)

        painter.setBrush(bg_color)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawEllipse(1, 1, 28, 28)

        pen = QPen(x_color, 2.2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        offset = 9.0
        painter.drawLine(offset, offset, 30.0 - offset, 30.0 - offset)
        painter.drawLine(30.0 - offset, offset, offset, 30.0 - offset)


class Live2DViewWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Live Sensor Stream (RGB-D)", parent)
        layout = QVBoxLayout(self)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #121214; border-radius: 4px;")
        self.image_label.setMinimumSize(424, 240)
        layout.addWidget(self.image_label)

    def set_theme_mode(self, is_dark: bool):
        bg = "#121214" if is_dark else "#e4e7ee"
        self.image_label.setStyleSheet(f"background-color: {bg}; border-radius: 4px;")

    def update_frame(self, bgr_image: np.ndarray):
        h, w, ch = bgr_image.shape
        bytes_per_line = ch * w
        rgb_img = np.ascontiguousarray(bgr_image[..., ::-1])
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)


class Viewport3DWidget(QGroupBox):
    """Hardware-accelerated 3D Viewport with external Header Grid toggle control."""
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("3D Model Inspection Viewport", parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 12, 6, 6)
        main_layout.setSpacing(6)

        # 1. Header Toolbar (Outside OpenGL canvas, inside GroupBox)
        header_bar = QHBoxLayout()
        header_bar.setContentsMargins(4, 0, 4, 0)
        
        header_spacer = QWidget()
        header_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header_bar.addWidget(header_spacer)

        # Grid Toggle Action Button
        self.grid_toggle_btn = QPushButton("⊞ Grid: ON")
        self.grid_toggle_btn.setCheckable(True)
        self.grid_toggle_btn.setChecked(True)
        self.grid_toggle_btn.setToolTip("Toggle 3D Floor Reference Grid")
        self.grid_toggle_btn.setFixedSize(90, 24)
        self.grid_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #23232a;
                color: #00ffcc;
                border: 1px solid #383844;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #2e2e38;
                border-color: #00ffcc;
            }
            QPushButton:checked {
                background-color: #00887a;
                color: #ffffff;
                border-color: #00ffcc;
            }
        """)
        self.grid_toggle_btn.toggled.connect(self._on_grid_toggled)
        header_bar.addWidget(self.grid_toggle_btn)
        main_layout.addLayout(header_bar)

        # 2. OpenGL Container
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_view = gl.GLViewWidget(self.container)
        self.gl_view.setBackgroundColor("#18181c")
        self.gl_view.setCameraPosition(pos=QVector3D(0, 0, 0), distance=1.5, elevation=28, azimuth=45)
        container_layout.addWidget(self.gl_view)

        # 3D Floor Grid Item
        self.grid = gl.GLGridItem()
        self.grid.setSize(2.5, 2.5, 2.5)
        self.grid.setSpacing(0.1, 0.1, 0.1)
        self.grid.setColor((100, 100, 120, 120))
        self.gl_view.addItem(self.grid)

        # Close Vector Button (Anchored overlay inside canvas)
        self.close_btn = CloseViewportButton(self.gl_view)
        self.close_btn.hide()
        self.close_btn.clicked.connect(self._on_close_clicked)

        main_layout.addWidget(self.container)

        self.is_dark_theme = True
        self.live_scatter_item = None
        self.current_mesh_item = None
        self.current_scatter_item = None

    def _on_grid_toggled(self, checked: bool):
        """Shows or hides the floor grid."""
        self.grid.setVisible(checked)
        self.grid_toggle_btn.setText("⊞ Grid: ON" if checked else "⊞ Grid: OFF")

    def set_theme_mode(self, is_dark: bool):
        self.is_dark_theme = is_dark
        if is_dark:
            self.gl_view.setBackgroundColor("#141418")
            self.grid.setColor((90, 95, 115, 100))
            self.grid_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #23232a;
                    color: #00ffcc;
                    border: 1px solid #383844;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background-color: #00887a;
                    color: #ffffff;
                }
            """)
        else:
            self.gl_view.setBackgroundColor("#3a3e47")
            self.grid.setColor((180, 185, 200, 120))
            self.grid_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #eaecf1;
                    color: #222226;
                    border: 1px solid #c2c6d0;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background-color: #00887a;
                    color: #ffffff;
                }
            """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.close_btn.move(self.gl_view.width() - 40, 10)

    def _on_close_clicked(self):
        self.clear_viewport()
        self.clear_requested.emit()

    def clear_viewport(self):
        if self.current_mesh_item is not None:
            self.gl_view.removeItem(self.current_mesh_item)
            self.current_mesh_item = None
        if self.current_scatter_item is not None:
            self.gl_view.removeItem(self.current_scatter_item)
            self.current_scatter_item = None
        if self.live_scatter_item is not None:
            self.gl_view.removeItem(self.live_scatter_item)
            self.live_scatter_item = None
        self.grid.resetTransform()
        self.close_btn.hide()

    def _correct_orientation(self, points: np.ndarray) -> np.ndarray:
        pts = np.empty_like(points, dtype=np.float32)
        pts[:, 0] = -points[:, 0]
        pts[:, 1] = -points[:, 2]
        pts[:, 2] = -points[:, 1]
        return pts

    def update_live_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        if len(points) == 0:
            return

        upright_pts = self._correct_orientation(points)
        alpha = np.ones((len(colors), 1), dtype=np.float32)
        rgba = np.hstack([np.clip(colors, 0.0, 1.0), alpha])

        if self.live_scatter_item is None:
            self.live_scatter_item = gl.GLScatterPlotItem(
                pos=upright_pts, color=rgba, size=2.5, pxMode=True
            )
            self.live_scatter_item.setGLOptions("opaque")
            self.gl_view.addItem(self.live_scatter_item)
        else:
            self.live_scatter_item.setData(pos=upright_pts, color=rgba)

    def display_mesh(self, vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray = None):
        self.clear_viewport()

        if len(vertices) == 0 or len(faces) == 0:
            return

        upright_verts = self._correct_orientation(vertices)
        corrected_faces = faces[:, [0, 2, 1]]

        min_bounds = np.min(upright_verts, axis=0)
        max_bounds = np.max(upright_verts, axis=0)
        center_xy = (min_bounds[:2] + max_bounds[:2]) / 2.0

        aligned_verts = upright_verts.copy()
        aligned_verts[:, 0] -= center_xy[0]
        aligned_verts[:, 1] -= center_xy[1]
        aligned_verts[:, 2] -= min_bounds[2]

        self.grid.resetTransform()
        self.grid.translate(0, 0, 0)

        bbox_diag = float(np.linalg.norm(max_bounds - min_bounds))
        cam_dist = max(0.8, bbox_diag * 1.5)
        z_center = float((max_bounds[2] - min_bounds[2]) * 0.4)

        self.gl_view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, z_center),
            distance=cam_dist,
            elevation=28,
            azimuth=45
        )

        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            rgba = np.zeros((len(vertices), 4), dtype=np.float32)
            rgba[:, :3] = np.clip(vertex_colors, 0.0, 1.0)
            rgba[:, 3] = 1.0

            mesh_data = gl.MeshData(
                vertexes=aligned_verts, faces=corrected_faces, vertexColors=rgba
            )
            self.current_mesh_item = gl.GLMeshItem(
                meshdata=mesh_data, smooth=False, drawEdges=False, shader=None, glOptions="opaque"
            )
        else:
            mesh_data = gl.MeshData(vertexes=aligned_verts, faces=corrected_faces)
            self.current_mesh_item = gl.GLMeshItem(
                meshdata=mesh_data, color=(0.3, 0.85, 0.95, 1.0),
                smooth=True, drawEdges=True, edgeColor=(0.15, 0.35, 0.45, 0.4),
                shader="shaded", glOptions="opaque"
            )

        self.gl_view.addItem(self.current_mesh_item)
        self.close_btn.show()
        self.close_btn.raise_()

    def display_point_cloud(self, points: np.ndarray, colors: np.ndarray = None):
        self.clear_viewport()

        if len(points) == 0:
            return

        upright_pts = self._correct_orientation(points)
        min_b = np.min(upright_pts, axis=0)
        max_b = np.max(upright_pts, axis=0)
        center_xy = (min_b[:2] + max_b[:2]) / 2.0

        aligned_pts = upright_pts.copy()
        aligned_pts[:, 0] -= center_xy[0]
        aligned_pts[:, 1] -= center_xy[1]
        aligned_pts[:, 2] -= min_b[2]

        self.grid.resetTransform()
        self.grid.translate(0, 0, 0)

        if colors is None or len(colors) != len(points):
            colors = np.array([[0.1, 0.85, 0.75, 1.0]] * len(points), dtype=np.float32)
        elif colors.shape[1] == 3:
            alpha = np.ones((len(points), 1), dtype=np.float32)
            colors = np.hstack([np.clip(colors, 0.0, 1.0), alpha])

        self.current_scatter_item = gl.GLScatterPlotItem(
            pos=aligned_pts, color=colors, size=2.5, pxMode=True
        )
        self.current_scatter_item.setGLOptions("opaque")
        self.gl_view.addItem(self.current_scatter_item)
        self.close_btn.show()
        self.close_btn.raise_()