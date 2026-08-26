# gui/models/reconstruction_workers.py
import os
import sys
import glob
import subprocess
import re
import shutil
import numpy as np
import open3d as o3d
from PySide6.QtCore import QObject, Signal, Slot


def glob_files(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.png"))
        + glob.glob(os.path.join(folder, "*.jpg"))
    )


class BaseReconstructionWorker(QObject):
    progress_changed = Signal(str, float)  # Stage, 0.0 - 1.0
    finished = Signal(str)                 # Output Mesh Path
    failed = Signal(str, str)              # Error Title, Error Message
    cancelled = Signal()                   # Cancellation confirmation

    def __init__(self, session_path: str, output_3d_dir: str):
        super().__init__()
        self.session_path = session_path
        self.output_3d_dir = output_3d_dir
        self._is_cancelled = False

    def request_cancel(self):
        """Thread-safe flag to request early worker termination."""
        self._is_cancelled = True


class Open3DTSDFWorker(BaseReconstructionWorker):
    """Volumetric TSDF integration with cooperative cancellation checks."""
    def __init__(self, session_path: str, output_3d_dir: str, voxel_size: float = 0.004):
        super().__init__(session_path, output_3d_dir)
        self.voxel_size = voxel_size

    @Slot()
    def run(self):
        try:
            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress_changed.emit("Loading Session Data...", 0.05)
            rgb_dir = os.path.join(self.session_path, "rgb")
            depth_dir = os.path.join(self.session_path, "depth")
            cam_k_path = os.path.join(self.session_path, "cam_K.txt")

            if not os.path.exists(cam_k_path):
                self.failed.emit("Missing Intrinsics", "cam_K.txt was not found in the dataset.")
                return

            cam_k = np.loadtxt(cam_k_path)
            width, height = 848, 480
            fx, fy, cx, cy = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2], cam_k[1, 2]
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size,
                sdf_trunc=self.voxel_size * 5.0,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            rgb_files = sorted(glob_files(rgb_dir))
            depth_files = sorted(glob_files(depth_dir))
            total_frames = len(rgb_files)

            if total_frames == 0:
                self.failed.emit("Empty Dataset", "No RGB images found to integrate.")
                return

            curr_pose = np.eye(4)
            prev_rgbd = None

            for idx, (rf, df) in enumerate(zip(rgb_files, depth_files)):
                # Periodic cancellation check during long frame processing
                if self._is_cancelled:
                    self.cancelled.emit()
                    return

                color = o3d.io.read_image(rf)
                depth = o3d.io.read_image(df)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=1.5, convert_rgb_to_intensity=False
                )

                if prev_rgbd is not None:
                    odo_success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                        rgbd, prev_rgbd, intrinsics, np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                        o3d.pipelines.odometry.OdometryOption()
                    )
                    if odo_success:
                        curr_pose = curr_pose @ trans

                volume.integrate(rgbd, intrinsics, np.linalg.inv(curr_pose))
                prev_rgbd = rgbd

                progress = 0.05 + (idx / total_frames) * 0.75
                self.progress_changed.emit(f"Integrating Frame {idx+1}/{total_frames}", progress)

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress_changed.emit("Extracting Polygon Mesh...", 0.85)
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()

            if self._is_cancelled:
                self.cancelled.emit()
                return

            os.makedirs(self.output_3d_dir, exist_ok=True)
            sess_name = os.path.basename(self.session_path)
            out_mesh_path = os.path.join(self.output_3d_dir, f"{sess_name}_tsdf.obj")
            o3d.io.write_triangle_mesh(out_mesh_path, mesh)

            self.progress_changed.emit("Complete!", 1.0)
            self.finished.emit(out_mesh_path)

        except Exception as e:
            self.failed.emit("TSDF Error", str(e))


class BundleSDFWorker(BaseReconstructionWorker):
    """Subprocess Neural Solver with active process killing upon user cancellation."""
    def __init__(self, session_path: str, output_3d_dir: str, project_root: str):
        super().__init__(session_path, output_3d_dir)
        self.project_root = project_root
        self._proc = None

    def request_cancel(self):
        """Terminates the active neural solver process immediately."""
        self._is_cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.kill()
            except Exception:
                pass

    @Slot()
    def run(self):
        try:
            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress_changed.emit("Verifying Session Assets...", 0.03)

            rgb_dir = os.path.join(self.session_path, "rgb")
            cam_k_path = os.path.join(self.session_path, "cam_K.txt")

            if not os.path.exists(cam_k_path):
                self.failed.emit("Dataset Incomplete", f"Missing cam_K.txt in {self.session_path}")
                return

            rgb_files = sorted(glob_files(rgb_dir))
            total_frames = max(1, len(rgb_files))

            mask_dir = os.path.join(self.session_path, "masks")
            if not os.path.exists(mask_dir) or len(os.listdir(mask_dir)) == 0:
                os.makedirs(mask_dir, exist_ok=True)
                import cv2
                if rgb_files:
                    sample = cv2.imread(rgb_files[0])
                    h, w = sample.shape[:2]
                    blank_mask = np.full((h, w), 255, dtype=np.uint8)
                    for rf in rgb_files:
                        fn = os.path.basename(rf)
                        cv2.imwrite(os.path.join(mask_dir, fn), blank_mask)

            self.progress_changed.emit("Configuring GPU Environment...", 0.06)

            torch_lib = subprocess.check_output(
                [sys.executable, "-c", "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
                text=True
            ).strip()
            btrack_build = os.path.join(self.project_root, "third_party", "bundlesdf", "BundleTrack", "build")

            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["PYTHONWARNINGS"] = "ignore"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["LD_LIBRARY_PATH"] = f"{torch_lib}:{btrack_build}:{env.get('LD_LIBRARY_PATH', '')}"
            env["PYTHONPATH"] = f"{self.project_root}/scripts:{self.project_root}/third_party/bundlesdf:{self.project_root}/third_party/bundlesdf/mycuda:{btrack_build}:{env.get('PYTHONPATH', '')}"

            out_folder = os.path.join(self.session_path, "output")
            os.makedirs(out_folder, exist_ok=True)
            solver_script = os.path.join(self.project_root, "third_party", "bundlesdf", "run_custom.py")

            self.progress_changed.emit("Launching Neural Solver...", 0.10)

            self._proc = subprocess.Popen(
                [
                    sys.executable, solver_script,
                    "--video_dir", self.session_path,
                    "--out_folder", out_folder,
                    "--use_segmenter", "0",
                    "--use_gui", "0"
                ],
                cwd=os.path.join(self.project_root, "third_party", "bundlesdf"),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_log = []
            for line in iter(self._proc.stdout.readline, ''):
                if self._is_cancelled:
                    self._proc.kill()
                    self.cancelled.emit()
                    return

                output_log.append(line)
                if len(output_log) > 40:
                    output_log.pop(0)

                # 1. Monotonic Frame Tracking Progress (10% -> 75%)
                if "process frame" in line:
                    m = re.search(r"process frame (\d+)", line)
                    if m:
                        curr_f = int(m.group(1))
                        progress = 0.10 + (curr_f / total_frames) * 0.65
                        self.progress_changed.emit(f"Tracking & Neural Mapping [{curr_f}/{total_frames}]", progress)

                # 2. Surface Mesh Extraction (75% -> 85%)
                elif "Running Marching Cubes" in line:
                    self.progress_changed.emit("Extracting Neural Implicit Mesh...", 0.85)

                # 3. High-Res UV Texture Baking (85% -> 98%)
                elif "project train_images" in line:
                    m = re.search(r"project train_images (\d+)/(\d+)", line)
                    if m:
                        cur_tex, tot_tex = int(m.group(1)), max(1, int(m.group(2)))
                        progress = 0.85 + (cur_tex / tot_tex) * 0.13
                        self.progress_changed.emit(f"Baking UV Texture [{cur_tex}/{tot_tex}]", progress)

            self._proc.wait()

            if self._is_cancelled:
                self.cancelled.emit()
                return

            if self._proc.returncode != 0:
                err_snippet = "".join(output_log[-12:]) if output_log else "No output from solver."
                self.failed.emit(
                    "Tracking Failure",
                    f"BundleSDF tracking failed or was lost during frame processing.\n\n"
                    f"Error Details:\n{err_snippet}"
                )
                return

            mesh_path = os.path.join(out_folder, "textured_mesh.obj")
            sess_name = os.path.basename(self.session_path)
            dest_obj = os.path.join(self.output_3d_dir, f"{sess_name}_bundlesdf.obj")

            if os.path.exists(mesh_path):
                shutil.copy(mesh_path, dest_obj)
                tex_src = os.path.join(out_folder, "textured_mesh.png")
                if os.path.exists(tex_src):
                    shutil.copy(tex_src, os.path.join(self.output_3d_dir, f"{sess_name}_bundlesdf.png"))
                mtl_src = os.path.join(out_folder, "textured_mesh.mtl")
                if os.path.exists(mtl_src):
                    shutil.copy(mtl_src, os.path.join(self.output_3d_dir, f"{sess_name}_bundlesdf.mtl"))

                self.progress_changed.emit("Done!", 1.0)
                self.finished.emit(dest_obj)
            else:
                self.failed.emit("Output Missing", "Reconstruction finished but textured_mesh.obj was not found.")

        except Exception as e:
            self.failed.emit("BundleSDF Worker Exception", str(e))