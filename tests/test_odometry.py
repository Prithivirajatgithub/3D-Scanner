import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c


WIDTH = 848
HEIGHT = 480
FPS = 30


# ------------------------------------------------------------
# RealSense
# ------------------------------------------------------------

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    WIDTH,
    HEIGHT,
    rs.format.z16,
    FPS
)

config.enable_stream(
    rs.stream.color,
    WIDTH,
    HEIGHT,
    rs.format.bgr8,
    FPS
)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

depth_sensor = (
    profile
    .get_device()
    .first_depth_sensor()
)

depth_scale = depth_sensor.get_depth_scale()

print(f"Depth scale = {depth_scale}")
print(f"Open3D depth_scale = {1.0 / depth_scale}")


# ------------------------------------------------------------
# Intrinsics
# ------------------------------------------------------------

color_profile = (
    profile
    .get_stream(rs.stream.color)
    .as_video_stream_profile()
)

intr = color_profile.get_intrinsics()

K = np.array(
    [
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ],
    dtype=np.float64
)

print("\nCamera matrix:")
print(K)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if o3c.cuda.is_available():
    device = o3c.Device("CUDA:0")
else:
    device = o3c.Device("CPU:0")

print(f"\nOpen3D device: {device}")


intrinsic = o3c.Tensor(
    K,
    dtype=o3c.float64,
    device=device
)


# ------------------------------------------------------------
# Capture first frame
# ------------------------------------------------------------

frames = pipeline.wait_for_frames()

frames = align.process(frames)

depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_np = np.asanyarray(
    depth_frame.get_data()
)

color_bgr = np.asanyarray(
    color_frame.get_data()
)

color_rgb = cv2.cvtColor(
    color_bgr,
    cv2.COLOR_BGR2RGB
)


print("\nFIRST FRAME")

print(
    "Depth:",
    depth_np.shape,
    depth_np.dtype
)

print(
    "Depth min:",
    depth_np[depth_np > 0].min()
    if np.any(depth_np > 0)
    else 0
)

print(
    "Depth max:",
    depth_np.max()
)

print(
    "Valid depth pixels:",
    np.count_nonzero(depth_np)
)


# ------------------------------------------------------------
# Open3D RGBD
# ------------------------------------------------------------

depth1 = o3d.t.geometry.Image(
    o3c.Tensor(
        depth_np,
        dtype=o3c.uint16,
        device=device
    )
)

color1 = o3d.t.geometry.Image(
    o3c.Tensor(
        color_rgb,
        dtype=o3c.uint8,
        device=device
    )
)

rgbd1 = o3d.t.geometry.RGBDImage(
    color1,
    depth1
)


print("\nMove the camera slightly.")

# ------------------------------------------------------------
# Second frame
# ------------------------------------------------------------

frames = pipeline.wait_for_frames()

frames = align.process(frames)

depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_np = np.asanyarray(
    depth_frame.get_data()
)

color_bgr = np.asanyarray(
    color_frame.get_data()
)

color_rgb = cv2.cvtColor(
    color_bgr,
    cv2.COLOR_BGR2RGB
)


depth2 = o3d.t.geometry.Image(
    o3c.Tensor(
        depth_np,
        dtype=o3c.uint16,
        device=device
    )
)

color2 = o3d.t.geometry.Image(
    o3c.Tensor(
        color_rgb,
        dtype=o3c.uint8,
        device=device
    )
)

rgbd2 = o3d.t.geometry.RGBDImage(
    color2,
    depth2
)


# ------------------------------------------------------------
# Odometry
# ------------------------------------------------------------

print("\nRunning RGB-D odometry...")

try:

    result = (
        o3d.t.pipelines.odometry
        .rgbd_odometry_multi_scale(
            rgbd1,
            rgbd2,
            intrinsic,

            o3c.Tensor(
                np.eye(4),
                dtype=o3c.float64,
                device=device
            ),

            depth_scale=1.0 / depth_scale,

            depth_max=3.0,

            method=(
                o3d.t.pipelines.odometry
                .Method.PointToPlane
            )
        )
    )

    print("\nRESULT")
    print("================================")

    print(
        "Fitness:",
        result.fitness
    )

    print(
        "RMSE:",
        result.inlier_rmse
    )

    print(
        "Transformation:"
    )

    print(
        result.transformation
    )

except Exception as e:

    print("\nODOMETRY ERROR:")
    print(e)


pipeline.stop()

print("\nTest finished.")
