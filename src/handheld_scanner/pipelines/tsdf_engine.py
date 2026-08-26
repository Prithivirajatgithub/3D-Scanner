import gc
import numpy as np
import open3d as o3d
from open3d.core import Tensor, Dtype


class TSDFEngine:
    def __init__(self, intrinsic_matrix, voxel_size=0.003, depth_max=1.2, device="CUDA:0"):
        self.device = o3d.core.Device(device if o3d.core.cuda.is_available() else "CPU:0")
        self.cpu_device = o3d.core.Device("CPU:0")
        self.voxel_size = voxel_size
        self.depth_max = depth_max
        self.trunc_multiplier = 2.5
        self.fitness_threshold = 0.12

        self.intrinsic_cpu = Tensor(intrinsic_matrix, Dtype.Float64, self.cpu_device)

        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(Dtype.Float32, Dtype.UInt16, Dtype.UInt16),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=16,
            block_count=30000,
            device=self.device,
        )

        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(15),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
        ]
        self.loss_params = o3d.t.pipelines.odometry.OdometryLossParams(
            depth_outlier_trunc=0.04,
            depth_huber_delta=0.02,
            intensity_huber_delta=0.1
        )

    def integrate(self, color_np, depth_np, pose_mat, depth_scale):
        color_int_t = o3d.t.geometry.Image(Tensor(color_np, device=self.device))
        depth_int_t = o3d.t.geometry.Image(Tensor(depth_np, device=self.device))
        extrinsic_t = Tensor(pose_mat, Dtype.Float64, self.cpu_device)

        block_coords = self.vbg.compute_unique_block_coordinates(
            depth_int_t, self.intrinsic_cpu, extrinsic_t,
            depth_scale=depth_scale, depth_max=self.depth_max, trunc_voxel_multiplier=self.trunc_multiplier
        )

        if block_coords.shape[0] > 0:
            self.vbg.integrate(
                block_coords, depth_int_t, color_int_t, self.intrinsic_cpu, extrinsic_t,
                depth_scale=depth_scale, depth_max=self.depth_max, trunc_voxel_multiplier=self.trunc_multiplier
            )
            return True
        return False

    def extract_point_cloud(self):
        try:
            t_pcd = self.vbg.extract_point_cloud()
            if t_pcd.point.positions.shape[0] > 0:
                return t_pcd.to_legacy()
        except RuntimeError:
            if o3d.core.cuda.is_available():
                o3d.core.cuda.release_cache()
            gc.collect()
        return None

    def extract_mesh(self):
        gc.collect()
        if o3d.core.cuda.is_available():
            o3d.core.cuda.release_cache()

        try:
            t_mesh = self.vbg.extract_triangle_mesh()
        except RuntimeError:
            vbg_cpu = self.vbg.to(self.cpu_device)
            t_mesh = vbg_cpu.extract_triangle_mesh()

        return t_mesh.to_legacy()
