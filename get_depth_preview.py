import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

sensor = pipeline_profile.get_device().query_sensors()[0]

sensor.set_option(rs.option.laser_power, 100)
sensor.set_option(rs.option.confidence_threshold, 1)
sensor.set_option(rs.option.min_distance, 0)
sensor.set_option(rs.option.enable_max_usable_range, 0)
sensor.set_option(rs.option.receiver_gain, 18)
sensor.set_option(rs.option.post_processing_sharpening, 3)
sensor.set_option(rs.option.pre_processing_sharpening, 5)
sensor.set_option(rs.option.noise_filtering, 6)

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

decimation_filter = rs.decimation_filter()
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        depth_frames = frames.get_depth_frame()

        depth_frames = decimation_filter.process(depth_frames)
        depth_frames = spatial_filter.process(depth_frames)
        depth_frames = temporal_filter.process(depth_frames)

        depth_image = np.asanyarray(depth_frames.get_data(), dtype=float)

        print(depth_image.shape)

        normalize = 10000

        depth_image = depth_image / normalize

        depth_image = np.dstack((depth_image, depth_image, depth_image))

        cv2.imshow('depth_frames', depth_image)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()