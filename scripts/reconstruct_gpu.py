import os
import sys
import glob
import json
import gc
import numpy as np
from tqdm import tqdm

# Force X11/XWayland backend for Open3D GUI
os.environ["WAYLAND_DISPLAY"] = ""

import open3d as o3d
from open3d.core import Tensor, Dtype

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct import get_latest_scan_dir, ask_export_format


def get_device():
    if o3d.core.cuda.is_available():
        print(f"[INFO] CUDA device detected -> using GPU ({o3d.core.cuda.device_count()} device(s))")
        return o3d.core.Device("CUDA:0")
    print("[WARN] No CUDA device found. Falling back to CPU tensor pipeline.")
    return o3d.core.Device("CPU:0")


def register_one_pair_gpu(t_rgbd_source, t_rgbd_target, intrinsic_t_cpu, device,
                           depth_scale=1000.0, depth_max=0.95, max_depth_diff=0.03):
    init_trans = Tensor(np.identity(4), Dtype.Float64, cpu_device)

    criteria_list = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
    ]

    loss_params = o3d.t.pipelines.odometry.OdometryLossParams(
        depth_outlier_trunc=max_depth_diff,
        depth_huber_delta=max_depth_diff * 0.5,
        intensity_huber_delta=0.1
    )

    # 1. Multi-scale Hybrid RGB-D Odometry (GPU)
    try:
        odo_result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            t_rgbd_source,
            t_rgbd_target,
            intrinsic_t_cpu,
            init_trans,
            depth_scale=depth_scale,
            depth_max=depth_max,
            criteria_list=criteria_list,
            method=o3d.t.pipelines.odometry.Method.Hybrid,
            params=loss_params,
        )
        
        if odo_result.fitness < 0.10:
            return False, np.identity(4), np.identity(6)
            
        trans_odo = odo_result.transformation
    except RuntimeError:
        return False, np.identity(4), np.identity(6)

    # 2. Point Cloud Creation & Safety Checks (GPU)
    pcd_source = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        t_rgbd_source, intrinsic_t_cpu, depth_scale=depth_scale, depth_max=depth_max
    )
    pcd_target = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        t_rgbd_target, intrinsic_t_cpu, depth_scale=depth_scale, depth_max=depth_max
    )

    if pcd_source.point.positions.shape[0] < 100 or pcd_target.point.positions.shape[0] < 100:
        return False, np.identity(4), np.identity(6)

    pcd_source = pcd_source.voxel_down_sample(voxel_size=0.005)
    pcd_target = pcd_target.voxel_down_sample(voxel_size=0.005)

    if pcd_source.point.positions.shape[0] < 50 or pcd_target.point.positions.shape[0] < 50:
        return False, np.identity(4), np.identity(6)

    pcd_source.estimate_normals()
    pcd_target.estimate_normals()

    # 3. Colored ICP Refinement (GPU)
    try:
        result_icp = o3d.t.pipelines.registration.icp(
            pcd_source,
            pcd_target,
            max_correspondence_distance=0.03,
            init_source_to_target=trans_odo,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            voxel_size=0.005,
        )

        if result_icp.fitness < 0.20:
            return False, np.identity(4), np.identity(6)

        info_t = o3d.t.pipelines.registration.get_information_matrix(
            pcd_source, pcd_target, 0.03, result_icp.transformation
        )
        return True, result_icp.transformation.cpu().numpy(), info_t.cpu().numpy()

    except RuntimeError:
        return False, np.identity(4), np.identity(6)


# Global CPU Device helper
cpu_device = o3d.core.Device("CPU:0")


def run_pipeline_gpu(scan_dir=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_base = os.path.join(project_root, "data", "raw")
    processed_base = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_base, exist_ok=True)

    if scan_dir is None:
        scan_dir = get_latest_scan_dir(raw_base)
        if scan_dir is None:
            print("[ERROR] No scan session found in data/raw/.")
            sys.exit(1)

    session_name = os.path.basename(scan_dir)
    export_formats = ask_export_format()
    device = get_device()

    intr_file = os.path.join(scan_dir, "camera_intrinsics.json")
    with open(intr_file, "r") as f:
        intr_data = json.load(f)

    intrinsic_np = np.array([
        [intr_data["fx"], 0, intr_data["cx"]],
        [0, intr_data["fy"], intr_data["cy"]],
        [0, 0, 1],
    ])
    # Intrinsic tensor MUST reside on CPU:0 for Open3D transformation bindings
    intrinsic_t_cpu = Tensor(intrinsic_np, Dtype.Float64, cpu_device)

    color_files = sorted(glob.glob(os.path.join(scan_dir, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(scan_dir, "depth", "*.png")))
    n_frames = min(len(color_files), len(depth_files))

    print(f"\n[INFO] Starting GPU Reconstruction: {session_name} ({n_frames} frame pairs)\n")

    depth_scale = 1000.0
    depth_max = 0.95

    # 1. Load RGB-D frames into SYSTEM RAM (CPU)
    cpu_rgbd_images = []
    for i in tqdm(range(n_frames), desc="[1/5] Reading RGB-D frames (Host RAM)", unit="frame"):
        color_t = o3d.t.io.read_image(color_files[i]).to(cpu_device)
        depth_t = o3d.t.io.read_image(depth_files[i]).to(cpu_device)
        cpu_rgbd_images.append(o3d.t.geometry.RGBDImage(color_t, depth_t))

    def get_gpu_pair(idx_a, idx_b):
        src = o3d.t.geometry.RGBDImage(
            cpu_rgbd_images[idx_a].color.to(device),
            cpu_rgbd_images[idx_a].depth.to(device)
        )
        tgt = o3d.t.geometry.RGBDImage(
            cpu_rgbd_images[idx_b].color.to(device),
            cpu_rgbd_images[idx_b].depth.to(device)
        )
        return src, tgt

    # 2. Sequential Odometry (GPU)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for s in tqdm(range(n_frames - 1), desc="[2/5] Sequential Odometry (GPU)", unit="pair"):
        t = s + 1
        src_gpu, tgt_gpu = get_gpu_pair(s, t)
        success, trans, info = register_one_pair_gpu(
            src_gpu, tgt_gpu, intrinsic_t_cpu, device,
            depth_scale=depth_scale, depth_max=depth_max,
        )
        del src_gpu, tgt_gpu

        if success:
            odometry = np.dot(odometry, trans)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))
        else:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))

    # 3. Strided Loop Closure Matching (GPU)
    stride = 10
    min_separation = 30
    candidate_pairs = [
        (s, t) for s in range(0, n_frames, stride) for t in range(s + min_separation, n_frames, stride)
    ]
    loop_closures_found = 0
    with tqdm(total=len(candidate_pairs), desc="[3/5] Loop Closure Matching (GPU)", unit="check") as pbar:
        for s, t in candidate_pairs:
            src_gpu, tgt_gpu = get_gpu_pair(s, t)
            success, trans, info = register_one_pair_gpu(
                src_gpu, tgt_gpu, intrinsic_t_cpu, device,
                depth_scale=depth_scale, depth_max=depth_max, max_depth_diff=0.05,
            )
            del src_gpu, tgt_gpu

            if success:
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))
                loop_closures_found += 1
            pbar.set_postfix({"loops_found": loop_closures_found})
            pbar.update(1)

    # 4. Global Optimization (CPU Levenberg-Marquardt)
    print("\n[INFO] Running Global Pose Graph Optimization...")
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option_opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option_opt)

    # Free cache before TSDF volume creation
    gc.collect()
    if device.get_type() == o3d.core.Device.DeviceType.CUDA:
        o3d.core.cuda.release_cache()

    # 5. GPU VoxelBlockGrid TSDF Integration
    print("\n[INFO] [4/5] GPU TSDF Volume Integration...")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(Dtype.Float32, Dtype.UInt16, Dtype.UInt16),
        attr_channels=((1), (1), (3)),
        voxel_size=0.003,
        block_resolution=16,
        block_count=20000,
        device=device,
    )

    for i in tqdm(range(len(cpu_rgbd_images)), desc="[4/5] TSDF Fusion (GPU)", unit="frame"):
        camera_pose = np.linalg.inv(pose_graph.nodes[i].pose)
        # Extrinsic tensor MUST remain on CPU:0 for VoxelBlockGrid geometry operations
        extrinsic_t_cpu = Tensor(camera_pose, Dtype.Float64, cpu_device)

        depth_i = cpu_rgbd_images[i].depth.to(device)
        color_i = cpu_rgbd_images[i].color.to(device)

        block_coords = vbg.compute_unique_block_coordinates(
            depth_i, intrinsic_t_cpu, extrinsic_t_cpu, depth_scale=depth_scale, depth_max=depth_max,
            trunc_voxel_multiplier=4.0
        )
        vbg.integrate(
            block_coords, depth_i, color_i, intrinsic_t_cpu, extrinsic_t_cpu,
            depth_scale=depth_scale, depth_max=depth_max, trunc_voxel_multiplier=4.0
        )
        del depth_i, color_i

    # Clean RAM & GPU Cache before extraction
    del cpu_rgbd_images
    gc.collect()
    if device.get_type() == o3d.core.Device.DeviceType.CUDA:
        o3d.core.cuda.release_cache()

    # Mesh extraction & Sharpening Cleanup
    print("\n[INFO] [5/5] Extracting triangle mesh & sharpening geometry...")
    try:
        t_mesh = vbg.extract_triangle_mesh()
    except RuntimeError:
        t_mesh = vbg.cpu().extract_triangle_mesh()
        
    mesh = t_mesh.to_legacy()

    # Base topology cleanup
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    # Isolate main object from floating noise specks
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    # Taubin smoothing to straighten planar edges
    mesh = mesh.filter_smooth_taubin(number_of_iterations=8, lambda_filter=0.5, mu=-0.53)
    mesh.compute_vertex_normals()

    print(f"\n=======================================================")
    print(f"[SUCCESS] GPU Mesh Generation Complete:")
    print(f"  -> Vertices: {len(mesh.vertices):,}")
    print(f"  -> Triangles: {len(mesh.triangles):,}")

    for fmt in export_formats:
        out_path = os.path.join(processed_base, f"{session_name}_mesh_gpu.{fmt}")
        if fmt == "stl":
            o3d.io.write_triangle_mesh(out_path, mesh)
        else:
            o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
        print(f"  -> Exported ({fmt.upper()}): {out_path}")
    print(f"=======================================================\n")

    o3d.visualization.draw_geometries(
        [mesh], window_name="3D Scanner Output Viewer (GPU)", width=1280, height=720
    )


if __name__ == "__main__":
    run_pipeline_gpu()