import os
import sys
import gc
import numpy as np

# Ensure X11 display connection
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

import open3d as o3d
from open3d.core import Tensor, Dtype

from handheld_scanner.core.camera_driver import RealSenseD456
from handheld_scanner.pipelines.tsdf_engine import TSDFEngine
from handheld_scanner.processing.mesh_cleaner import MeshProcessor


def main():
    print("=" * 65)
    print("       HANDHELD 3D SCANNER - MEMORY-SAFE LIVE PIPELINE")
    print("=" * 65)

    # 1. Initialize RealSense Camera Driver
    cam = RealSenseD456(width=848, height=480, fps=30, enable_emitter=True)
    intrinsics_np = cam.get_intrinsics_matrix()
    depth_scale = cam.depth_scale

    # 2. Initialize TSDF Engine
    engine = TSDFEngine(
        intrinsic_matrix=intrinsics_np,
        voxel_size=0.003,
        depth_max=1.2,
        device="CUDA:0"
    )

    # 3. Viewport Setup
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Handheld 3D Scanner - Live Session", width=1280, height=720)

    is_running = True
    is_scanning = True

    def stop_cb(vis_obj):
        nonlocal is_running
        is_running = False
        return False

    def toggle_pause_cb(vis_obj):
        nonlocal is_scanning
        is_scanning = not is_scanning
        status = "RECORDING" if is_scanning else "PAUSED"
        print(f"\n[SCANNER STATE] -> {status}")
        return False

    vis.register_key_callback(ord('Q'), stop_cb)
    vis.register_key_callback(ord('q'), stop_cb)
    vis.register_key_callback(27, stop_cb)
    vis.register_key_callback(32, toggle_pause_cb)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    vis.add_geometry(coord_frame)

    current_pcd_geom = None
    first_view_reset = False

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.asarray([0.08, 0.08, 0.08])
        opt.point_size = 2.5

    prev_rgbd = None
    curr_pose = np.identity(4)
    integrated_frames = 0
    render_interval = 4  # Refresh live 3D mesh view every 4th frame to protect VRAM

    print("\n--- Controls ---")
    print("  [SPACE]      : PAUSE / RESUME scanning")
    print("  [Q] / [ESC]  : FINISH scan and generate 3D mesh")
    print("  Rotate scene : Left-Click + Drag inside window")
    print("  Pan scene    : Shift + Left-Click + Drag")
    print("  Zoom         : Scroll Wheel\n")
    print("[STATUS] Live 3D Reconstruction: ACTIVE")

    try:
        while is_running:
            color_np, depth_np = cam.get_frame(apply_filters=True)
            if color_np is None or depth_np is None:
                continue

            color_f = (color_np.astype(np.float32) / 255.0)
            depth_f = depth_np.astype(np.float32) / depth_scale

            color_t = o3d.t.geometry.Image(Tensor(color_f, device=engine.device))
            depth_t = o3d.t.geometry.Image(Tensor(depth_f, device=engine.device))
            curr_rgbd = o3d.t.geometry.RGBDImage(color_t, depth_t)

            valid_pixels = np.count_nonzero((depth_np > 0) & (depth_np < int(engine.depth_max * depth_scale)))

            if is_scanning and valid_pixels >= 1500:
                if prev_rgbd is None:
                    if engine.integrate(color_np, depth_np, curr_pose, depth_scale):
                        prev_rgbd = curr_rgbd
                        integrated_frames += 1
                else:
                    init_trans = Tensor(np.identity(4), Dtype.Float64, engine.cpu_device)
                    try:
                        odo_result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                            prev_rgbd,
                            curr_rgbd,
                            engine.intrinsic_cpu,
                            init_trans,
                            depth_scale=1.0,
                            depth_max=engine.depth_max,
                            criteria_list=engine.criteria_list,
                            method=o3d.t.pipelines.odometry.Method.Hybrid,
                            params=engine.loss_params,
                        )

                        if odo_result.fitness >= engine.fitness_threshold:
                            delta_trans = odo_result.transformation.cpu().numpy()
                            curr_pose = np.dot(curr_pose, delta_trans)
                            prev_rgbd = curr_rgbd

                            if engine.integrate(color_np, depth_np, curr_pose, depth_scale):
                                integrated_frames += 1

                            print(f"\r[SCANNING] Frames: {integrated_frames} | Fitness: {odo_result.fitness:.2f}", end="")
                    except RuntimeError:
                        pass
            elif not is_scanning:
                print(f"\r[STATUS: PAUSED] Press SPACEBAR to Resume...", end="")

            # Live Viewport Rendering with memory management
            if integrated_frames == 1 or (integrated_frames > 1 and integrated_frames % render_interval == 0):
                pcd_legacy = engine.extract_point_cloud()
                if pcd_legacy is not None:
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

            # Periodic CUDA cache release every 50 frames
            if integrated_frames > 0 and integrated_frames % 50 == 0:
                if o3d.core.cuda.is_available():
                    o3d.core.cuda.release_cache()
                gc.collect()

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\n\n[INFO] Scan stopped by user.")

    finally:
        cam.stop()
        vis.destroy_window()

    # 4. Extract, Clean & Export
    if integrated_frames < 3:
        print(f"\n[ERROR] Only {integrated_frames} frames integrated. Insufficient geometric data.")
        sys.exit(1)

    print("\n\n[INFO] Extracting mesh from TSDF volume...")
    raw_mesh = engine.extract_mesh()

    print("[INFO] Cleaning mesh geometry...")
    cleaned_mesh = MeshProcessor.clean_mesh(raw_mesh, min_cluster_size=300, smooth_iterations=6)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "live_recordings")

    obj_p, stl_p = MeshProcessor.export(cleaned_mesh, output_dir, prefix="live_scan")

    print(f"\n=======================================================")
    print(f"[SUCCESS] Exported Reconstructed 3D Meshes:")
    print(f"  -> OBJ (Colored) : {obj_p}")
    print(f"  -> STL (CAD-Ready): {stl_p}")
    print(f"  -> Vertices      : {len(cleaned_mesh.vertices):,}")
    print(f"  -> Triangles     : {len(cleaned_mesh.triangles):,}")
    print(f"=======================================================\n")

    vis_final = o3d.visualization.Visualizer()
    vis_final.create_window(window_name="Final 3D Reconstructed Model", width=1280, height=720)
    vis_final.add_geometry(cleaned_mesh)
    ctr_final = vis_final.get_view_control()
    ctr_final.set_up([0, -1, 0])
    ctr_final.set_front([0, 0, -1])
    vis_final.run()
    vis_final.destroy_window()


if __name__ == "__main__":
    main()
