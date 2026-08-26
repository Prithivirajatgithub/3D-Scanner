# gui/controllers/scanner_controller.py
import os
import shutil
import numpy as np
import open3d as o3d
from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtWidgets import QFileDialog

from gui.views.main_window import MainWindow
from gui.views.export_dialog import Export3DDialog
from gui.models.session_manager import SessionManager, ScanSession
from gui.models.reconstruction_workers import Open3DTSDFWorker, BundleSDFWorker
from gui.models.camera_streamer import CameraStreamerWorker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "bundlesdf_recordings")
OUTPUT_3D_DIR = os.path.join(RECORDINGS_DIR, "3D_output_files")


class ScannerController(QObject):
    def __init__(self, view: MainWindow):
        super().__init__()
        self.view = view
        self.session_manager = SessionManager(RECORDINGS_DIR)

        self.current_mesh = None
        self.current_mesh_path = None

        self.recon_thread = None
        self.recon_worker = None
        self.camera_thread = None
        self.camera_worker = None

        # Connect UI signals
        self.view.scan_triggered.connect(self.on_scan_triggered)
        self.view.export_requested.connect(self.on_export_requested)
        self.view.open_mesh_requested.connect(self.on_open_mesh_requested)
        self.view.view_3d.clear_requested.connect(self.on_viewport_cleared)
        self.view.control_panel.refresh_btn.clicked.connect(self.refresh_sessions)
        self.view.control_panel.session_selected.connect(self.on_session_selected)
        self.view.control_panel.reconstruct_requested.connect(self.on_run_reconstruction)
        self.view.control_panel.open_folder_requested.connect(self.on_browse_folder)

        self.refresh_sessions()

    def refresh_sessions(self, select_session_path: str = None):
        sessions = self.session_manager.list_sessions()
        self.view.control_panel.set_sessions(sessions)
        if select_session_path:
            for i in range(self.view.control_panel.session_combo.count()):
                if self.view.control_panel.session_combo.itemData(i) == select_session_path:
                    self.view.control_panel.session_combo.setCurrentIndex(i)
                    break

    @Slot(str)
    def on_session_selected(self, session_path: str):
        sess = ScanSession(session_path)
        k_str = "Loaded" if sess.cam_k is not None else "Missing"
        self.view.control_panel.session_info_lbl.setText(
            f"Frames: {sess.num_frames} | Intrinsics: {k_str}"
        )

    @Slot()
    def on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self.view, "Select Scan Session Directory", RECORDINGS_DIR
        )
        if folder:
            sess = ScanSession(folder)
            if sess.is_valid:
                self.view.control_panel.set_sessions([sess])

    @Slot(str, str, dict)
    def on_run_reconstruction(self, session_path: str, algo_name: str, params: dict):
        self.view.control_panel.run_recon_btn.setEnabled(False)
        self.view.save_btn.setEnabled(False)
        self.view.control_panel.set_active_device_state("CPU")
        self.recon_thread = QThread()

        if "Open3D" in algo_name:
            self.recon_worker = Open3DTSDFWorker(
                session_path, OUTPUT_3D_DIR, params.get("voxel_size", 0.004)
            )
        else:
            self.recon_worker = BundleSDFWorker(
                session_path, OUTPUT_3D_DIR, PROJECT_ROOT
            )

        self.recon_worker.moveToThread(self.recon_thread)
        self.recon_thread.started.connect(self.recon_worker.run)
        self.recon_worker.progress_changed.connect(self.on_recon_progress)
        self.recon_worker.finished.connect(self.on_recon_finished)
        self.recon_worker.failed.connect(self.on_recon_failed)

        self.recon_thread.start()

    @Slot(str, float)
    def on_recon_progress(self, stage: str, pct: float):
        self.view.control_panel.status_lbl.setText(f"Status: {stage}")
        self.view.control_panel.progress_bar.setValue(int(pct * 100))
        

    @Slot(str)
    def on_recon_finished(self, mesh_path: str):
        self.view.control_panel.run_recon_btn.setEnabled(True)
        self.view.control_panel.set_active_device_state(False)
    
        if self.recon_thread:
            self.recon_thread.quit()
            self.recon_thread.wait()

        self.view.status_bar.showMessage(
            f"Reconstruction Complete: {os.path.basename(mesh_path)}"
        )
        self._load_and_display_mesh(mesh_path)

    def _load_and_display_mesh(self, file_path: str):
        """Loads OBJ/STL/PLY/PCD and renders directly in the 3D inspection viewport."""
        if not os.path.exists(file_path):
            self.view.show_error_popup("File Error", f"Cannot find file:\n{file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        self.current_mesh_path = file_path

        try:
            if ext in [".obj", ".stl", ".ply"]:
                mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=True)
                mesh.compute_vertex_normals()
                self.current_mesh = mesh
                self.view.save_btn.setEnabled(True)

                verts = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.triangles, dtype=np.int32)
                colors = (
                    np.asarray(mesh.vertex_colors, dtype=np.float32)
                    if mesh.has_vertex_colors()
                    else None
                )

                if len(faces) > 0:
                    self.view.view_3d.display_mesh(verts, faces, colors)
                elif len(verts) > 0:
                    self.view.view_3d.display_point_cloud(verts, colors)

            elif ext == ".pcd":
                pcd = o3d.io.read_point_cloud(file_path)
                pts = np.asarray(pcd.points, dtype=np.float32)
                colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
                
                # Wrap point cloud into dummy mesh for export capability
                dummy_mesh = o3d.geometry.TriangleMesh()
                dummy_mesh.vertices = pcd.points
                if pcd.has_colors():
                    dummy_mesh.vertex_colors = pcd.colors
                self.current_mesh = dummy_mesh
                self.view.save_btn.setEnabled(True)

                self.view.view_3d.display_point_cloud(pts, colors)

            self.view.status_bar.showMessage(f"Opened 3D Model: {os.path.basename(file_path)}")
        except Exception as e:
            self.view.show_error_popup("Read Error", f"Failed to open 3D model:\n\n{str(e)}")

    @Slot()
    def on_open_mesh_requested(self):
        """File browser to pick and load any existing 3D file."""
        filter_str = (
            "All Supported 3D Models (*.obj *.stl *.ply *.pcd);;"
            "Wavefront 3D Mesh (*.obj);;"
            "Stereolithography CAD (*.stl);;"
            "Polygon File Format (*.ply);;"
            "Point Cloud Data (*.pcd)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self.view, "Open 3D Model", OUTPUT_3D_DIR, filter_str
        )
        if file_path:
            self._load_and_display_mesh(file_path)

    @Slot()
    def on_viewport_cleared(self):
        self.current_mesh = None
        self.current_mesh_path = None
        self.view.save_btn.setEnabled(False)
        self.view.status_bar.showMessage("Viewport Cleared | Ready")

    @Slot(str, str)
    def on_recon_failed(self, title: str, msg: str):
        self.view.control_panel.run_recon_btn.setEnabled(True)
        self.view.control_panel.set_active_device_state(False)
        
        if self.recon_thread:
            self.recon_thread.quit()
            self.recon_thread.wait()
        self.view.show_error_popup(title, msg)

    @Slot()
    def on_export_requested(self):
        if self.current_mesh is None or len(self.current_mesh.vertices) == 0:
            self.view.show_error_popup("Export Error", "No active 3D model available to export.")
            return

        default_name = os.path.splitext(os.path.basename(self.current_mesh_path or "reconstructed_mesh"))[0]
        dialog = Export3DDialog(OUTPUT_3D_DIR, default_name, parent=self.view)
        if dialog.exec():
            save_path, ext = dialog.get_selected_target()
            try:
                export_mesh = o3d.geometry.TriangleMesh(self.current_mesh)
                export_mesh.compute_vertex_normals()

                if ext in [".obj", ".ply"]:
                    o3d.io.write_triangle_mesh(
                        save_path,
                        export_mesh,
                        write_ascii=False,
                        compressed=False,
                        write_vertex_normals=True,
                        write_vertex_colors=True,
                    )
                    if ext == ".obj" and self.current_mesh_path:
                        src_dir = os.path.dirname(self.current_mesh_path)
                        for companion in [".png", ".mtl"]:
                            src_c = os.path.splitext(self.current_mesh_path)[0] + companion
                            if os.path.exists(src_c):
                                dst_c = os.path.splitext(save_path)[0] + companion
                                shutil.copy(src_c, dst_c)

                elif ext == ".stl":
                    o3d.io.write_triangle_mesh(
                        save_path,
                        export_mesh,
                        write_ascii=False,
                    )

                elif ext == ".pcd":
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = export_mesh.vertices
                    pcd.normals = export_mesh.vertex_normals
                    if export_mesh.has_vertex_colors():
                        pcd.colors = export_mesh.vertex_colors
                    o3d.io.write_point_cloud(save_path, pcd)

                self.view.status_bar.showMessage(f"Saved: {save_path}")
                self.view.show_info_popup(
                    "Export Successful",
                    f"Your 3D model was exported to:\n\n{save_path}"
                )
            except Exception as e:
                self.view.show_error_popup("Export Failed", f"Could not export 3D file:\n\n{str(e)}")

    @Slot(bool)
    def on_scan_triggered(self, start_scan: bool):
        if start_scan:
            self.view.view_3d.clear_viewport()
            self.camera_thread = QThread()
            self.camera_worker = CameraStreamerWorker(record_dir_base=RECORDINGS_DIR)
            self.camera_worker.moveToThread(self.camera_thread)

            self.camera_thread.started.connect(self.camera_worker.start_streaming)
            self.camera_worker.frame_received.connect(self.on_frame_received)
            self.camera_worker.pointcloud_received.connect(self.on_pointcloud_received)
            self.camera_worker.camera_error.connect(self.on_camera_error)
            self.camera_worker.camera_connected.connect(self.on_camera_connected)
            self.camera_worker.scan_saved.connect(self.on_scan_saved)

            self.camera_thread.start()
        else:
            if self.camera_worker:
                self.camera_worker.stop_streaming()
            if self.camera_thread:
                self.camera_thread.quit()
                self.camera_thread.wait(2000)

    @Slot(np.ndarray, np.ndarray)
    def on_frame_received(self, color_bgr: np.ndarray, depth_u16: np.ndarray):
        self.view.view_2d.update_frame(color_bgr)

    @Slot(np.ndarray, np.ndarray)
    def on_pointcloud_received(self, points: np.ndarray, colors: np.ndarray):
        self.view.view_3d.update_live_point_cloud(points, colors)

    @Slot(str, int)
    def on_scan_saved(self, session_dir: str, frame_count: int):
        self.view.status_bar.showMessage(f"Scan Completed: {frame_count} frames recorded.")
        self.refresh_sessions(select_session_path=session_dir)
        self.view.show_info_popup(
            "Scan Recorded Successfully",
            f"Capture session saved successfully!\n\n"
            f"• Total Frames: {frame_count}\n"
            f"• Location: {session_dir}\n\n"
            f"The session is now selected and ready for 3D reconstruction."
        )

    @Slot(str, str)
    def on_camera_error(self, title: str, message: str):
        self.on_scan_triggered(False)
        self.view.scan_btn.blockSignals(True)
        self.view.scan_btn.setChecked(False)
        self.view.scan_btn.setText("START SCAN")
        self.view.scan_btn.blockSignals(False)

        self.view.status_bar.showMessage(f"Error: {title}")
        self.view.show_error_popup(title, message)

    @Slot(str)
    def on_camera_connected(self, dev_name: str):
        self.view.status_bar.showMessage(f"Recording Live: {dev_name} (30 FPS)")