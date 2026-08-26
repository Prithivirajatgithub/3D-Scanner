# gui/models/session_manager.py
import os
import glob
import numpy as np

class ScanSession:
    """Represents a recorded RGB-D dataset."""
    def __init__(self, session_path: str):
        self.path = os.path.abspath(session_path)
        self.name = os.path.basename(self.path)
        self.rgb_dir = os.path.join(self.path, "rgb")
        self.depth_dir = os.path.join(self.path, "depth")
        self.mask_dir = os.path.join(self.path, "masks")
        self.cam_k_path = os.path.join(self.path, "cam_K.txt")
        self.output_dir = os.path.join(self.path, "output")
        
        self.num_frames = len(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        self.cam_k = None
        if os.path.exists(self.cam_k_path):
            try:
                self.cam_k = np.loadtxt(self.cam_k_path)
            except Exception:
                pass

    @property
    def is_valid(self) -> bool:
        return os.path.isdir(self.rgb_dir) and self.num_frames > 0


class SessionManager:
    """Manages discovery and loading of scanning sessions."""
    def __init__(self, base_recordings_dir: str):
        self.base_dir = os.path.abspath(base_recordings_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def list_sessions(self) -> list[ScanSession]:
        sessions = []
        if not os.path.exists(self.base_dir):
            return sessions
        for entry in sorted(os.listdir(self.base_dir), reverse=True):
            full_path = os.path.join(self.base_dir, entry)
            if os.path.isdir(full_path) and entry.startswith("scan_"):
                sess = ScanSession(full_path)
                if sess.is_valid:
                    sessions.append(sess)
        return sessions