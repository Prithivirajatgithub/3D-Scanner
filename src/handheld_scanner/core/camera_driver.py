import numpy as np
import pyrealsense2 as rs


class RealSenseD456:
    def __init__(self, width=848, height=480, fps=30, enable_emitter=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_emitter = enable_emitter

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = 1.0 / self.depth_sensor.get_depth_scale()

        # IR Projector Settings
        if self.enable_emitter and self.depth_sensor.supports(rs.option.emitter_enabled):
            self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if self.enable_emitter and self.depth_sensor.supports(rs.option.laser_power):
            l_range = self.depth_sensor.get_option_range(rs.option.laser_power)
            self.depth_sensor.set_option(rs.option.laser_power, l_range.max)

        # Fast Post-Processing Depth Filters
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        self.hole_filling = rs.hole_filling_filter(1)

        # Intrinsic Matrix
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()

    def get_intrinsics_matrix(self):
        return np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def get_frame(self, apply_filters=True):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        d_frame = aligned.get_depth_frame()
        c_frame = aligned.get_color_frame()

        if not d_frame or not c_frame:
            return None, None

        if apply_filters:
            d_frame = self.spatial.process(d_frame)
            d_frame = self.temporal.process(d_frame)
            d_frame = self.hole_filling.process(d_frame)

        color_img = np.ascontiguousarray(c_frame.get_data())
        depth_img = np.ascontiguousarray(d_frame.get_data())
        return color_img, depth_img

    def stop(self):
        self.pipeline.stop()
