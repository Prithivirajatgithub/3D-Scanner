import os
import glob
import json
import sys
import numpy as np
from tqdm import tqdm

# Force X11/XWayland backend for Open3D GUI
os.environ["WAYLAND_DISPLAY"] = ""

import open3d as o3d


def get_latest_scan_dir(raw_dir):
    sessions = sorted(glob.glob(os.path.join(raw_dir, "scan_*")))
    if not sessions:
        return None
    return sessions[-1]


def ask_export_format():
    print("\nSelect 3D Mesh Output Format:")
    print("  [1] STL (.stl) -> Geometry only (3D Printing / CAD)")
    print("  [2] OBJ (.obj) -> 3D Mesh with RGB Colors (MeshLab / Blender / Unreal)")
    print("  [3] PLY (.ply) -> Polygon Mesh with Per-Vertex Color")
    print("  [4] ALL        -> Export STL, OBJ, and PLY simultaneously")

    choice = input("\nEnter choice [1/2/3/4] (Default: 2): ").strip()
    if choice == "1":
        return ["stl"]
    elif choice == "3":
        return ["ply"]
    elif choice == "4":
        return ["stl", "obj", "ply"]
    else:
        return ["obj"]


def register_one_pair(rgbd_source, rgbd_target, intrinsic, init_trans, max_depth_diff=0.03):
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = max_depth_diff

    # 1. Fast Coarse RGB-D Odometry
    success_odo, trans_odo, info_odo = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_source,
        rgbd_target,
        intrinsic,
        init_trans,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    if not success_odo:
        return False, np.identity(4), np.identity(6)

    # 2. Point Cloud Creation & Safety Checks
    pcd_source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_source, intrinsic)
    pcd_target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_target, intrinsic)

    # Check for empty/sparse point clouds to avoid KDTreeFlann warnings
    if len(pcd_source.points) < 100 or len(pcd_target.points) < 100:
        return False, np.identity(4), np.identity(6)

    pcd_source = pcd_source.voxel_down_sample(voxel_size=0.005)
    pcd_target = pcd_target.voxel_down_sample(voxel_size=0.005)

    if len(pcd_source.points) < 50 or len(pcd_target.points) < 50:
        return False, np.identity(4), np.identity(6)

    pcd_source.estimate_normals()
    pcd_target.estimate_normals()

    # 3. Safe Colored ICP Refinement
    try:
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            pcd_source,
            pcd_target,
            max_correspondence_distance=0.03,
            init=trans_odo,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )

        # 0.20 threshold balances rotation tracking and false-match rejection
        if result_icp.fitness < 0.15: #0.20
            return False, np.identity(4), np.identity(6)

        info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcd_source, pcd_target, 0.03, result_icp.transformation
        )
        return True, result_icp.transformation, info_matrix

    except RuntimeError:
        return False, np.identity(4), np.identity(6)


def run_pipeline(scan_dir=None):
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

    intr_file = os.path.join(scan_dir, "camera_intrinsics.json")
    with open(intr_file, "r") as f:
        intr_data = json.load(f)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr_data["width"],
        intr_data["height"],
        intr_data["fx"],
        intr_data["fy"],
        intr_data["cx"],
        intr_data["cy"],
    )

    color_files = sorted(glob.glob(os.path.join(scan_dir, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(scan_dir, "depth", "*.png")))
    n_frames = min(len(color_files), len(depth_files))

    print(f"\n[INFO] Starting Reconstruction: {session_name} ({n_frames} frame pairs)\n")

    # 1. Load RGB-D frames (Balanced depth cutoff at 0.80 m)
    rgbd_images = []
    for i in tqdm(range(n_frames), desc="[1/5] Loading RGB-D frames", unit="frame"):
        color = o3d.io.read_image(color_files[i])
        depth = o3d.io.read_image(depth_files[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=1.8,  #0.80 old value
            convert_rgb_to_intensity=False,
        )
        rgbd_images.append(rgbd)

    # 2. Sequential Odometry
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for s in tqdm(range(n_frames - 1), desc="[2/5] Sequential Odometry", unit="pair"):
        t = s + 1
        success, trans, info = register_one_pair(
            rgbd_images[s], rgbd_images[t], intrinsic, np.identity(4)
        )

        if success:
            odometry = np.dot(odometry, trans)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=False)
            )
        else:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
            )

    # 3. Strided Loop Closure Candidate Sampling
    stride = 10
    min_separation = 30
    candidate_pairs = [
        (s, t)
        for s in range(0, n_frames, stride)
        for t in range(s + min_separation, n_frames, stride)
    ]
    loop_closures_found = 0

    with tqdm(
        total=len(candidate_pairs), desc="[3/5] Loop Closure Matching", unit="check"
    ) as pbar:
        for s, t in candidate_pairs:
            success, trans, info = register_one_pair(
                rgbd_images[s],
                rgbd_images[t],
                intrinsic,
                np.identity(4),
                max_depth_diff=0.05,
            )
            if success:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True)
                )
                loop_closures_found += 1
            pbar.set_postfix({"loops_found": loop_closures_found})
            pbar.update(1)

    # 4. Global Optimization
    print("\n[INFO] Running Global Pose Graph Optimization...")
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option_opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph, method, criteria, option_opt
    )

    # 5. TSDF Volume Integration
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.0015,#003 - gives details for the object decrease the value for details
        sdf_trunc=0.012,#0.012
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in tqdm(range(len(rgbd_images)), desc="[4/5] TSDF Volume Integration", unit="frame"):
        camera_pose = np.linalg.inv(pose_graph.nodes[i].pose)
        volume.integrate(rgbd_images[i], intrinsic, camera_pose)

    # Mesh extraction & cleanup
    print("\n[INFO] [5/5] Extracting triangle mesh & cleaning geometry...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    print(f"\n=======================================================")
    print(f"[SUCCESS] Mesh Generation Complete:")
    print(f"  -> Vertices: {len(mesh.vertices):,}")
    print(f"  -> Triangles: {len(mesh.triangles):,}")

    for fmt in export_formats:
        out_path = os.path.join(processed_base, f"{session_name}_mesh.{fmt}")
        if fmt == "stl":
            o3d.io.write_triangle_mesh(out_path, mesh)
        else:
            o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
        print(f"  -> Exported ({fmt.upper()}): {out_path}")
    print(f"=======================================================\n")

    # Native interactive preview
    o3d.visualization.draw_geometries(
        [mesh], window_name="3D Scanner Output Viewer", width=1280, height=720
    )


if __name__ == "__main__":
    run_pipeline()