# gui/models/camera_streamer.py
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtCore import QObject, Signal, Slot


class CameraStreamerWorker(QObject):
    frame_received = Signal(np.ndarray, np.ndarray)        # color_bgr, depth_u16
    pointcloud_received = Signal(np.ndarray, np.ndarray)   # points_xyz, colors_rgb
    camera_error = Signal(str, str)                        # title, message
    camera_connected = Signal(str)                         # device name
    scan_saved = Signal(str, int)                          # session_dir, frame_count

    def __init__(self, record_dir_base: str = None):
        super().__init__()
        self.is_running = False
        self.record_dir_base = record_dir_base
        self.is_recording = False
        self.pipeline = None
        self.align = None
        self.session_dir = None
        self.frame_idx = 0

    def check_device_connected(self) -> bool:
        ctx = rs.context()
        return len(ctx.query_devices()) > 0

    @Slot()
    def start_streaming(self):
        if not self.check_device_connected():
            self.camera_error.emit(
                "Hardware Not Found",
                "No Intel RealSense camera detected.\n\nPlease plug your RealSense D400-series USB 3.0 cable and try again."
            )
            return

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            width, height, fps = 848, 480, 30

            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            dev_name = profile.get_device().get_info(rs.camera_info.name)
            self.camera_connected.emit(dev_name)

            # Retrieve Intrinsics
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

            # Prepare recording directory if base path provided
            if self.record_dir_base:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.session_dir = os.path.join(self.record_dir_base, f"scan_{timestamp}")
                self.color_dir = os.path.join(self.session_dir, "rgb")
                self.depth_dir = os.path.join(self.session_dir, "depth")
                self.mask_dir = os.path.join(self.session_dir, "masks")
                os.makedirs(self.color_dir, exist_ok=True)
                os.makedirs(self.depth_dir, exist_ok=True)
                os.makedirs(self.mask_dir, exist_ok=True)

                cam_k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
                np.savetxt(os.path.join(self.session_dir, "cam_K.txt"), cam_k, fmt="%.6f")
                self.is_recording = True
                self.frame_idx = 0

            self.is_running = True
            
            # Pre-compute pixel grid for fast point cloud extraction
            u, v = np.meshgrid(np.arange(0, width, 4), np.arange(0, height, 4))
            u_flat = u.flatten()
            v_flat = v.flatten()

            while self.is_running:
                frames = self.pipeline.wait_for_frames(timeout_ms=3000)
                aligned = self.align.process(frames)
                depth_f = aligned.get_depth_frame()
                color_f = aligned.get_color_frame()

                if not depth_f or not color_f:
                    continue

                color_np = np.ascontiguousarray(color_f.get_data())
                depth_np = np.ascontiguousarray(depth_f.get_data(), dtype=np.uint16)

                # 1. Emit 2D frame
                self.frame_received.emit(color_np, depth_np)

                # 2. Record frame if recording is active
                if self.is_recording:
                    fn = f"{self.frame_idx:06d}.png"
                    cv2.imwrite(os.path.join(self.color_dir, fn), color_np)
                    cv2.imwrite(os.path.join(self.depth_dir, fn), depth_np)
                    # Create full valid mask
                    mask = np.full((height, width), 255, dtype=np.uint8)
                    cv2.imwrite(os.path.join(self.mask_dir, fn), mask)
                    self.frame_idx += 1

                # 3. Unproject live point cloud for real-time 3D display
                depth_sampled = depth_np[v_flat, u_flat].astype(np.float32) / 1000.0  # to meters
                valid = (depth_sampled > 0.15) & (depth_sampled < 2.5)

                if np.any(valid):
                    z = depth_sampled[valid]
                    x = (u_flat[valid] - cx) * z / fx
                    y = (v_flat[valid] - cy) * z / fy
                    
                    pts = np.stack([x, y, z], axis=-1)
                    rgb_sampled = color_np[v_flat[valid], u_flat[valid], ::-1].astype(np.float32) / 255.0

                    self.pointcloud_received.emit(pts, rgb_sampled)

        except Exception as e:
            self.is_running = False
            self.camera_error.emit("Camera Pipeline Error", str(e))
        finally:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None

            if self.is_recording and self.session_dir and self.frame_idx > 0:
                self.scan_saved.emit(self.session_dir, self.frame_idx)
                self.is_recording = False

    @Slot()
    def stop_streaming(self):
        self.is_running = False