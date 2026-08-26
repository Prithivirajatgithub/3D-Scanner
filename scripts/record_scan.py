import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import time
import threading
import queue
from datetime import datetime

def writer_worker(write_queue):
    """Background worker that handles blocking disk I/O."""
    while True:
        item = write_queue.get()
        if item is None:
            break
        c_path, c_img, d_path, d_img = item
        cv2.imwrite(c_path, c_img)
        cv2.imwrite(d_path, d_img)
        write_queue.task_done()

def create_scan_session(base_data_dir="data/raw"):
    session_id = datetime.now().strftime("scan_%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_data_dir, session_id)
    color_dir = os.path.join(session_dir, "color")
    depth_dir = os.path.join(session_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    return session_dir, color_dir, depth_dir

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_raw_dir = os.path.join(project_root, "data", "raw")
    session_dir, color_dir, depth_dir = create_scan_session(data_raw_dir)

    pipeline = rs.pipeline()
    config = rs.config()

    # 1. 848x480 @ 60 FPS provides a high frame rate with minimal sensor noise
    WIDTH, HEIGHT, FPS = 848, 480, 30
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    intrinsic_dict = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": intrinsics.coeffs
    }

    with open(os.path.join(session_dir, "camera_intrinsics.json"), "w") as f:
        json.dump(intrinsic_dict, f, indent=4)

    # 2. Async queue to prevent disk write stalls
    write_queue = queue.Queue(maxsize=300)
    worker = threading.Thread(target=writer_worker, args=(write_queue,), daemon=True)
    worker.start()

    print(f"[INFO] Scan directory initialized at: {session_dir}")
    print("\n--- Controls ---")
    print("SPACEBAR : Start / Pause Recording")
    print("Q / ESC  : Stop and Save\n")

    frame_count = 0
    recording = False
    loop_count = 0

    fps_timer = time.time()
    current_fps = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Enqueue copy of frames to keep buffer valid
            if recording:
                filename = f"{frame_count:06d}.png"
                c_path = os.path.join(color_dir, filename)
                d_path = os.path.join(depth_dir, filename)
                try:
                    write_queue.put_nowait((c_path, color_image.copy(), d_path, depth_image.copy()))
                    frame_count += 1
                except queue.Full:
                    print("[WARN] Disk write buffer full! Dropping frame.")

            # 3. Update preview only every 2nd frame to save CPU
            loop_count += 1
            if loop_count % 2 == 0:
                # Downsample before color mapping
                small_depth = cv2.resize(depth_image, (424, 240), interpolation=cv2.INTER_NEAREST)
                small_color = cv2.resize(color_image, (424, 240), interpolation=cv2.INTER_NEAREST)

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(small_depth, alpha=0.04), cv2.COLORMAP_JET
                )
                preview = np.hstack((small_color, depth_colormap))

                if loop_count % 30 == 0:
                    now = time.time()
                    current_fps = 30 / (now - fps_timer)
                    fps_timer = now

                status_text = f"FPS: {current_fps:.1f} | REC: {'ON' if recording else 'PAUSED'} | Saved: {frame_count}"
                status_color = (0, 0, 255) if recording else (0, 255, 0)
                cv2.putText(preview, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                cv2.imshow("Handheld 3D Scanner - Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                recording = not recording
                print(f"[STATUS] Recording {'STARTED' if recording else 'PAUSED'}")
            elif key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Finishing disk writes ({write_queue.qsize()} remaining)...")
        write_queue.put(None)
        worker.join()
        print(f"[INFO] Capture closed. Total frames saved: {frame_count} in {session_dir}")

if __name__ == "__main__":
    main()