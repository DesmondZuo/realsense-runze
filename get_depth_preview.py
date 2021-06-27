import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())

        depth_image = np.dstack((depth_image, depth_image, depth_image))

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #cv2.namedWindow('depth_frames', cv2.WINDOW_NORMAL)
        cv2.imshow('depth_frames', depth_colormap)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()