import os
import sys
import time
import gc
import numpy as np
import pyrealsense2 as rs

# Ensure X11 display connection on Linux/Ubuntu
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

import open3d as o3d
from open3d.core import Tensor, Dtype


def main():
    print("=" * 65)
    print("  INTEL REALSENSE D456 INDUSTRIAL 3D SCANNER (MODEL-TO-FRAME)")
    print("=" * 65)

    # 1. GPU Compute Setup
    if o3d.core.cuda.is_available():
        device = o3d.core.Device("CUDA:0")
        print("[INFO] Compute Engine       : GPU (NVIDIA CUDA)")
    else:
        device = o3d.core.Device("CPU:0")
        print("[WARN] Compute Engine       : CPU (Fallback)")
    cpu_device = o3d.core.Device("CPU:0")

    # 2. Configure RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    width, height, target_fps = 848, 480, 30

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, target_fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, target_fps)

    # Enable Built-in IMU Sensors
    imu_available = False
    try:
        config.enable_stream(rs.stream.gyro)
        config.enable_stream(rs.stream.accel)
        imu_available = True
    except Exception:
        imu_available = False

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Configure Laser Projector & Power
    depth_sensor = profile.get_device().first_depth_sensor()
    ir_status = "Disabled"
    laser_power = 0.0

    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        ir_status = "ENABLED (Active IR Dot Pattern)"

    if depth_sensor.supports(rs.option.laser_power):
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_range.max)
        laser_power = laser_range.max

    # Hardware Depth Filters
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter(1)

    # Camera Intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    depth_scale_val = 1.0 / depth_sensor.get_depth_scale()

    intrinsic_np = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ])
    intrinsic_t_cpu = Tensor(intrinsic_np, Dtype.Float64, cpu_device)
    intrinsic_t_gpu = Tensor(intrinsic_np, Dtype.Float64, device)

    # Warmup
    print("[INFO] Stabilizing camera sensors...")
    for _ in range(15):
        frames = pipeline.wait_for_frames()
        align.process(frames)

    print("\n" + "-" * 65)
    print("                 DEVICE STATUS & DIAGNOSTICS")
    print("-" * 65)
    print(f"  Camera Model      : Intel RealSense D456")
    print(f"  Resolution & Rate : {width}x{height} @ {target_fps} FPS")
    print(f"  IR Dot Projector  : {ir_status} (Power: {laser_power} mW)")
    print(f"  Built-in 6DoF IMU : {'ENABLED (Active)' if imu_available else 'DISABLED'}")
    print(f"  Focal Parameters  : fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}")
    print("-" * 65 + "\n")

    # 3. TSDF Volume Configuration (Optimal for objects & tabletops)
    voxel_size = 0.004          # 4 mm voxel size for crisp, low-noise surfaces
    depth_min = 0.25            # 25 cm min range
    depth_max = 1.00            # 1.0 m max range (prevents background noise)
    trunc_multiplier = 4.0       # Stable surface band (16 mm)
    fitness_threshold = 0.20

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(Dtype.Float32, Dtype.UInt16, Dtype.UInt16),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=40000,
        device=device,
    )

    criteria_list = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
    ]
    loss_params = o3d.t.pipelines.odometry.OdometryLossParams(
        depth_outlier_trunc=0.03,
        depth_huber_delta=0.015,
        intensity_huber_delta=0.1
    )

    # 4. Viewport Setup
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="RealSense 3D Scanner - Model-to-Frame SLAM", width=1280, height=720, visible=True)

    is_running = True
    is_scanning = True

    def stop_callback(vis_obj):
        nonlocal is_running
        is_running = False
        return False

    def toggle_pause_callback(vis_obj):
        nonlocal is_scanning
        is_scanning = not is_scanning
        state_str = "RECORDING" if is_scanning else "PAUSED"
        print(f"\n[SCANNER] -> {state_str}")
        return False

    vis.register_key_callback(ord('Q'), stop_callback)
    vis.register_key_callback(ord('q'), stop_callback)
    vis.register_key_callback(27, stop_callback)
    vis.register_key_callback(32, toggle_pause_callback)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    vis.add_geometry(coord_frame)

    current_pcd_geom = None
    first_view_reset = False

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.asarray([0.08, 0.08, 0.08])
        opt.point_size = 2.5

    curr_pose = np.identity(4)
    model_rgbd = None
    integrated_frames = 0

    print("--- Controls ---")
    print("  [SPACE]     : PAUSE / RESUME scanning")
    print("  [Q] / [ESC] : FINISH scanning and generate 3D mesh")
    print("  [Mouse]     : Left-Click to Rotate, Shift+Click to Pan, Scroll to Zoom\n")
    print("[STATUS] Live 3D Reconstruction: ACTIVE")

    try:
        while is_running:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Apply Depth Filters
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color_np = np.ascontiguousarray(color_frame.get_data())
            depth_np = np.ascontiguousarray(depth_frame.get_data())

            # Range Gate: Clip room background
            depth_np[(depth_np < int(depth_min * depth_scale_val)) | (depth_np > int(depth_max * depth_scale_val))] = 0

            color_f = (color_np.astype(np.float32) / 255.0)
            depth_f = depth_np.astype(np.float32) / depth_scale_val

            color_t = o3d.t.geometry.Image(Tensor(color_f, device=device))
            depth_t = o3d.t.geometry.Image(Tensor(depth_f, device=device))
            curr_rgbd = o3d.t.geometry.RGBDImage(color_t, depth_t)

            color_int_t = o3d.t.geometry.Image(Tensor(color_np, device=device))
            depth_int_t = o3d.t.geometry.Image(Tensor(depth_np, device=device))

            valid_pixels = np.count_nonzero(depth_np > 0)

            if is_scanning and valid_pixels >= 4000:
                if integrated_frames == 0:
                    # Initial Frame Integration
                    extrinsic_t_cpu = Tensor(curr_pose, Dtype.Float64, cpu_device)
                    block_coords = vbg.compute_unique_block_coordinates(
                        depth_int_t, intrinsic_t_cpu, extrinsic_t_cpu,
                        depth_scale=depth_scale_val, depth_max=depth_max, trunc_voxel_multiplier=trunc_multiplier
                    )
                    if block_coords.shape[0] > 0:
                        vbg.integrate(
                            block_coords, depth_int_t, color_int_t, intrinsic_t_cpu, extrinsic_t_cpu,
                            depth_scale=depth_scale_val, depth_max=depth_max, trunc_voxel_multiplier=trunc_multiplier
                        )
                        model_rgbd = curr_rgbd
                        integrated_frames += 1
                else:
                    # Model-to-Frame Odometry Alignment (Eliminates Drift)
                    init_trans = Tensor(np.identity(4), Dtype.Float64, cpu_device)
                    try:
                        odo_result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                            curr_rgbd,
                            model_rgbd,
                            intrinsic_t_cpu,
                            init_trans,
                            depth_scale=1.0,
                            depth_max=depth_max,
                            criteria_list=criteria_list,
                            method=o3d.t.pipelines.odometry.Method.Hybrid,
                            params=loss_params,
                        )

                        if odo_result.fitness >= fitness_threshold:
                            delta_trans = odo_result.transformation.cpu().numpy()
                            curr_pose = np.dot(curr_pose, np.linalg.inv(delta_trans))

                            extrinsic_t_cpu = Tensor(curr_pose, Dtype.Float64, cpu_device)
                            block_coords = vbg.compute_unique_block_coordinates(
                                depth_int_t, intrinsic_t_cpu, extrinsic_t_cpu,
                                depth_scale=depth_scale_val, depth_max=depth_max, trunc_voxel_multiplier=trunc_multiplier
                            )
                            if block_coords.shape[0] > 0:
                                vbg.integrate(
                                    block_coords, depth_int_t, color_int_t, intrinsic_t_cpu, extrinsic_t_cpu,
                                    depth_scale=depth_scale_val, depth_max=depth_max, trunc_voxel_multiplier=trunc_multiplier
                                )
                                integrated_frames += 1

                                # Raycast model to update synthetic reference every 3 integrated frames
                                if integrated_frames % 3 == 0:
                                    extrinsic_t_gpu = Tensor(curr_pose, Dtype.Float64, device)
                                    raycast_res = vbg.ray_cast(
                                        block_coords=vbg.get_unique_block_coordinates(),
                                        intrinsic=intrinsic_t_gpu,
                                        extrinsic=extrinsic_t_gpu,
                                        width=width,
                                        height=height,
                                        render_attributes=["depth", "color"],
                                        depth_scale=depth_scale_val,
                                        depth_max=depth_max,
                                        trunc_voxel_multiplier=trunc_multiplier
                                    )
                                    model_rgbd = o3d.t.geometry.RGBDImage(
                                        raycast_res["color"].to(Dtype.Float32) / 255.0,
                                        raycast_res["depth"] / depth_scale_val
                                    )

                            print(f"\r[STATUS: SCANNING] Keyframes: {integrated_frames} | Fit: {odo_result.fitness:.2f}", end="")
                    except RuntimeError:
                        pass

            # 5. Live Point Cloud Stream
            if integrated_frames > 0:
                try:
                    t_pcd = vbg.extract_point_cloud()
                    if t_pcd.point.positions.shape[0] > 0:
                        pcd_legacy = t_pcd.to_legacy()

                        if current_pcd_geom is not None:
                            vis.remove_geometry(current_pcd_geom, reset_bounding_box=False)

                        current_pcd_geom = pcd_legacy
                        vis.add_geometry(current_pcd_geom, reset_bounding_box=not first_view_reset)

                        if not first_view_reset:
                            vis.reset_view_point(True)
                            ctr = vis.get_view_control()
                            ctr.set_up([0, -1, 0])
                            ctr.set_front([0, 0, -1])
                            ctr.set_lookat([0, 0, 0.45])
                            first_view_reset = True
                except RuntimeError:
                    pass

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\n\n[INFO] Scan stopped by user.")

    finally:
        pipeline.stop()
        vis.destroy_window()

    # 6. Post-Processing & Export
    if integrated_frames < 3:
        print(f"\n[ERROR] Only {integrated_frames} frames integrated. Insufficient geometric data.")
        sys.exit(1)

    print("\n\n[INFO] Extracting 3D triangle mesh from TSDF volume...")

    del model_rgbd, curr_rgbd, color_t, depth_t, color_int_t, depth_int_t
    gc.collect()
    if o3d.core.cuda.is_available():
        o3d.core.cuda.release_cache()

    try:
        t_mesh = vbg.extract_triangle_mesh()
    except RuntimeError:
        print("[WARN] Offloading to CPU for Marching Cubes...")
        vbg_cpu = vbg.to(cpu_device)
        del vbg
        gc.collect()
        if o3d.core.cuda.is_available():
            o3d.core.cuda.release_cache()
        t_mesh = vbg_cpu.extract_triangle_mesh()

    mesh = t_mesh.to_legacy()

    print("[INFO] Cleaning mesh geometry & surface normals...")
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    # Filter isolated noise islands
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    triangles_to_remove = np.zeros(len(mesh.triangles), dtype=bool)
    for c_idx, count in enumerate(cluster_n_triangles):
        if count < 500:
            triangles_to_remove |= (triangle_clusters == c_idx)

    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    mesh = mesh.filter_smooth_taubin(number_of_iterations=8, lambda_filter=0.5, mu=-0.53)
    mesh.compute_vertex_normals()

    # Save to data/live_Recordings/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "live_Recordings")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_obj = os.path.join(output_dir, f"live_scan_{timestamp}.obj")
    out_stl = os.path.join(output_dir, f"live_scan_{timestamp}.stl")

    o3d.io.write_triangle_mesh(out_obj, mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(out_stl, mesh)

    print(f"\n=======================================================")
    print(f"[SUCCESS] Exported Reconstructed 3D Meshes:")
    print(f"  -> OBJ (Colored) : {out_obj}")
    print(f"  -> STL (CAD-Ready): {out_stl}")
    print(f"  -> Vertices      : {len(mesh.vertices):,}")
    print(f"  -> Triangles     : {len(mesh.triangles):,}")
    print(f"=======================================================\n")

    vis_final = o3d.visualization.Visualizer()
    vis_final.create_window(window_name="Final 3D Reconstructed Model", width=1280, height=720)
    vis_final.add_geometry(mesh)
    ctr_final = vis_final.get_view_control()
    ctr_final.set_up([0, -1, 0])
    ctr_final.set_front([0, 0, -1])
    vis_final.run()
    vis_final.destroy_window()


if __name__ == "__main__":
    main()