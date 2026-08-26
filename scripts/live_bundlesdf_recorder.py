#!/usr/bin/env python3
import glob
import os
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

PROJECT_ROOT = os.path.expanduser("~/projects/handheld-3d-scanner")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "bundlesdf_recordings")
OUTPUT_3D_DIR = os.path.join(RECORDINGS_DIR, "3D_output_files")

os.makedirs(OUTPUT_3D_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)


class ROISelector:

  def __init__(self, window_name):
    self.window_name = window_name
    self.roi = None
    self.drawing = False
    self.start_pt = None
    self.current_pt = None

  def mouse_callback(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.drawing = True
      self.start_pt = (x, y)
      self.current_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
      self.current_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
      self.drawing = False
      x1, y1 = self.start_pt
      x2, y2 = x, y
      xmin, xmax = min(x1, x2), max(x1, x2)
      ymin, ymax = min(y1, y2), max(y1, y2)
      if (xmax - xmin) > 15 and (ymax - ymin) > 15:
        self.roi = (xmin, ymin, xmax - xmin, ymax - ymin)


def capture_session():
  pipeline = rs.pipeline()
  config = rs.config()
  width, height, fps = 848, 480, 30

  config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
  config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

  profile = pipeline.start(config)
  align = rs.align(rs.stream.color)

  depth_sensor = profile.get_device().first_depth_sensor()
  if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
  if depth_sensor.supports(rs.option.laser_power):
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, laser_range.max)

  spatial = rs.spatial_filter()
  spatial.set_option(rs.option.filter_magnitude, 2)
  spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
  temporal = rs.temporal_filter()
  hole_filling = rs.hole_filling_filter(1)

  color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
  intr = color_stream.get_intrinsics()

  timestamp = time.strftime("%Y%m%d_%H%M%S")
  dataset_dir = os.path.join(RECORDINGS_DIR, f"scan_{timestamp}")
  color_dir = os.path.join(dataset_dir, "rgb")
  depth_dir = os.path.join(dataset_dir, "depth")
  mask_dir = os.path.join(dataset_dir, "masks")

  os.makedirs(color_dir, exist_ok=True)
  os.makedirs(depth_dir, exist_ok=True)
  os.makedirs(mask_dir, exist_ok=True)

  cam_k = np.array(
      [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
  )
  np.savetxt(os.path.join(dataset_dir, "cam_K.txt"), cam_k, fmt="%.6f")

  window_name = "BundleSDF Target Selector: Drag Box -> Press SPACE"
  cv2.namedWindow(window_name)
  selector = ROISelector(window_name)
  cv2.setMouseCallback(window_name, selector.mouse_callback)

  print(
      "\n[STEP 1] Drag a box over target object, then press SPACEBAR to lock"
      " selection."
  )
  first_color = None

  while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_f = aligned.get_depth_frame()
    color_f = aligned.get_color_frame()
    if not depth_f or not color_f:
      continue

    color_img = np.ascontiguousarray(color_f.get_data())
    display = color_img.copy()

    if selector.drawing and selector.start_pt and selector.current_pt:
      cv2.rectangle(
          display, selector.start_pt, selector.current_pt, (0, 255, 0), 2
      )
    elif selector.roi:
      x, y, w, h = selector.roi
      cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(window_name, display)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and selector.roi is not None:
      first_color = color_img.copy()
      break
    elif key == 27 or key == ord("q"):
      pipeline.stop()
      cv2.destroyAllWindows()
      shutil.rmtree(dataset_dir, ignore_errors=True)
      return None

  cv2.destroyWindow(window_name)

  rx, ry, rw, rh = selector.roi
  init_mask = np.zeros((height, width), dtype=np.uint8)
  init_mask[ry : ry + rh, rx : rx + rw] = 255

  record_window = "BundleSDF Active Scan [Orbit camera | Q/ESC: Finish]"
  cv2.namedWindow(record_window)
  frame_idx = 0
  print("\n[STEP 2] Orbit smoothly around object.")
  print("  -> Press [Q] or [ESC] when completed.\n")

  try:
    while True:
      frames = pipeline.wait_for_frames()
      aligned = align.process(frames)
      depth_f = aligned.get_depth_frame()
      color_f = aligned.get_color_frame()
      if not depth_f or not color_f:
        continue

      depth_f = spatial.process(depth_f)
      depth_f = temporal.process(depth_f)
      depth_f = hole_filling.process(depth_f)

      color_np = np.ascontiguousarray(color_f.get_data())
      depth_np = np.ascontiguousarray(depth_f.get_data(), dtype=np.uint16)

      curr_mask = init_mask.copy()

      fn = f"{frame_idx:06d}.png"
      cv2.imwrite(os.path.join(color_dir, fn), color_np)
      cv2.imwrite(os.path.join(depth_dir, fn), depth_np)
      cv2.imwrite(os.path.join(mask_dir, fn), curr_mask)
      frame_idx += 1

      hud = color_np.copy()
      cv2.rectangle(hud, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
      cv2.putText(
          hud,
          f"RECORDING Frame: {frame_idx:04d}",
          (20, 35),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
      )
      cv2.putText(
          hud,
          "Press 'Q' or ESC to Finish Scanning",
          (20, 70),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (255, 255, 255),
          1,
      )

      cv2.imshow(record_window, hud)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q") or key == 27:
        break
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()

  print(
      f"\n[+] Captured {frame_idx} aligned RGB-D-Mask frames in: {dataset_dir}"
  )
  return dataset_dir


def run_reconstruction(dataset_dir):
  print("\n" + "=" * 50)
  print("  SELECT 3D OUTPUT FORMAT TO EXPORT")
  print("=" * 50)
  print("  [1] OBJ (.obj textured mesh + material)")
  print("  [2] STL (.stl CAD/3D Printing surface)")
  print("  [3] Point Cloud (.ply & .pcd dense point cloud)")
  print("  [4] ALL FORMATS (.obj, .stl, .ply, .pcd) [DEFAULT]")
  choice = input("Enter choice [1/2/3/4] (default: 4): ").strip()
  if choice not in ["1", "2", "3", "4"]:
    choice = "4"

  torch_lib = subprocess.check_output(
      [
          sys.executable,
          "-c",
          (
              "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"
              " 'lib'))"
          ),
      ],
      text=True,
  ).strip()
  btrack_build = os.path.join(
      PROJECT_ROOT, "third_party", "bundlesdf", "BundleTrack", "build"
  )

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = "0"
  env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  env["LD_LIBRARY_PATH"] = (
      f"{torch_lib}:{btrack_build}:{env.get('LD_LIBRARY_PATH', '')}"
  )
  env["PYTHONPATH"] = (
      f"{PROJECT_ROOT}/scripts:{PROJECT_ROOT}/third_party/bundlesdf:{PROJECT_ROOT}/third_party/bundlesdf/mycuda:{btrack_build}:{env.get('PYTHONPATH', '')}"
  )

  out_folder = os.path.join(dataset_dir, "output")
  os.makedirs(out_folder, exist_ok=True)

  solver_script = os.path.join(
      PROJECT_ROOT, "third_party", "bundlesdf", "run_custom.py"
  )
  print(f"\n[*] Launching BundleSDF Neural Solver: {solver_script}")

  subprocess.run(
      [
          sys.executable,
          solver_script,
          "--video_dir",
          dataset_dir,
          "--out_folder",
          out_folder,
          "--use_segmenter",
          "0",
          "--use_gui",
          "0",
      ],
      env=env,
      check=True,
  )

  mesh_path = os.path.join(out_folder, "textured_mesh.obj")
  if not os.path.exists(mesh_path):
    candidates = glob.glob(os.path.join(out_folder, "*.obj")) + glob.glob(
        os.path.join(out_folder, "*.ply")
    )
    candidates = [
        m for m in candidates if not m.endswith("cloud_for_init_coord.ply")
    ]
    if candidates:
      mesh_path = candidates[0]
    else:
      print("\n[!] No reconstructed 3D mesh was found in the output folder.")
      return

  scan_name = os.path.basename(dataset_dir)
  mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
  mesh.compute_vertex_normals()

  pcd = o3d.geometry.PointCloud()
  pcd.points = mesh.vertices
  pcd.normals = mesh.vertex_normals
  if mesh.has_vertex_colors():
    pcd.colors = mesh.vertex_colors

  saved_paths = []

  # Export OBJ + Texture Map
  if choice in ["1", "4"]:
    dest_obj = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_mesh.obj")
    o3d.io.write_triangle_mesh(dest_obj, mesh)
    saved_paths.append(dest_obj)

    tex_src = os.path.join(out_folder, "textured_mesh.png")
    if os.path.exists(tex_src):
      dest_png = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_mesh.png")
      shutil.copy(tex_src, dest_png)
      saved_paths.append(dest_png)

    mtl_src = os.path.join(out_folder, "textured_mesh.mtl")
    if os.path.exists(mtl_src):
      dest_mtl = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_mesh.mtl")
      shutil.copy(mtl_src, dest_mtl)
      saved_paths.append(dest_mtl)

  # Export STL (CAD / 3D Printing)
  if choice in ["2", "4"]:
    dest_stl = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_model.stl")
    o3d.io.write_triangle_mesh(dest_stl, mesh)
    saved_paths.append(dest_stl)

  # Export Point Cloud (PLY & PCD)
  if choice in ["3", "4"]:
    dest_pcd = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_cloud.pcd")
    dest_ply = os.path.join(OUTPUT_3D_DIR, f"{scan_name}_cloud.ply")
    o3d.io.write_point_cloud(dest_pcd, pcd)
    o3d.io.write_point_cloud(dest_ply, pcd)
    saved_paths.append(dest_pcd)
    saved_paths.append(dest_ply)

  print("\n" + "=" * 65)
  print(f"[SUCCESS] 3D Model Assets Exported to: {OUTPUT_3D_DIR}")
  for path in saved_paths:
    print(f"  -> {path}")
  print("=" * 65)

  o3d.visualization.draw_geometries(
      [mesh], window_name=f"Reconstructed Mesh: {scan_name}"
  )


def main():
  while True:
    print("\n" + "=" * 65)
    print("   HANDHELD 3D SCANNER - LIVE BUNDLESDF INTERACTIVE SUITE")
    print("=" * 65)

    dataset_dir = capture_session()
    if dataset_dir is None:
      print("[INFO] Capture aborted.")
      break

    print("\n" + "-" * 50)
    print(" SCAN COMPLETED. CHOOSE ACTION:")
    print("-" * 50)
    print("  [1] Run 3D Neural Reconstruction (BundleSDF) [DEFAULT]")
    print("  [2] Rescan (Discard current session & scan again)")
    print("  [3] Exit")

    action = input("\nEnter choice [1/2/3] (default: 1): ").strip()
    if action not in ["1", "2", "3"]:
      action = "1"

    if action == "1":
      run_reconstruction(dataset_dir)
      break
    elif action == "2":
      print(f"[*] Discarding session: {dataset_dir}")
      shutil.rmtree(dataset_dir, ignore_errors=True)
      print("[*] Restarting scanner...")
      time.sleep(1)
      continue
    else:
      print("[*] Exiting.")
      break


if __name__ == "__main__":
  main()