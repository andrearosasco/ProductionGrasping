import os
import time

import cv2
import pyrealsense2 as rs

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.io import loadmat


class RealSense:
    def __init__(self, width=640, heigth=480):
        self.pipeline = rs.pipeline()
        configs = {}
        configs['device'] = 'Intel RealSense D435i'

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.profile = self.pipeline.start(config)

        configs['depth'] = {'width': width, 'height': heigth, 'format': 'z16', 'fps': 30}
        configs['color'] = {'width': width, 'height': heigth, 'format': 'rgb8', 'fps': 30}

        HIGH_ACCURACY = 3
        HIGH_DENSITY = 4
        MEDIUM_DENSITY = 5
        self.profile.get_device().sensors[0].set_option(rs.option.visual_preset, HIGH_DENSITY)

        # sensors = self.profile.get_device().query_sensors()
        # for sensor in sensors:
        #     if sensor.supports(rs.option.auto_exposure_priority):
        #         exp = sensor.set_option(rs.option.auto_exposure_priority, 0)
        #         exp = sensor.get_option(rs.option.auto_exposure_priority)

        configs['options'] = {}
        for device in self.profile.get_device().sensors:
            configs['options'][device.name] = {}
            for option in device.get_supported_options():
                configs['options'][device.name][str(option)[7:]] = str(device.get_option(option))

        self.configs = configs
        self.align = rs.align(rs.stream.color)

    def intrinsics(self):
        return self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        # color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    @classmethod
    def pointcloud(cls, depth_image, rgb_image=None):
        if rgb_image is None:
            return cls._pointcloud(depth_image)

        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=1000)



        # intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375,
        #               'ppy': 237.75723266601562,
        #               'width': 640, 'height': 480}

        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}

        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    @classmethod
    def _pointcloud(cls, depth_image):
        depth_image = o3d.geometry.Image(depth_image)

        # intrinsics = {'fx': 384.025146484375, 'fy': 384.025146484375, 'ppx': 319.09661865234375, 'ppy': 237.75723266601562,
        #               'width': 640, 'height': 480}

        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}

        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'],
                                                   intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'])

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return np.array(pcd.points)

    def stop(self):
        self.pipeline.stop()


class RealSenseCamera:

    def run(self):
        i = 0
        camera = RealSense()
        while (True):
            frame, depth = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Solo quando leggiamo con cv2
            #
            # frame = np.array(Image.open('test3_rgb.png'))
            # depth = np.array(Image.open('test3_depth.png')).astype(np.uint16)

            cv2.imshow('view', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1)
            if key == ord('c'):
                timer = 3
                elapsed, count = 0, 0
                start = time.time()
                while elapsed <= timer + 1:
                    frame, depth = camera.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow('view', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                    elapsed = time.time() - start

                    for t in range(timer + 1):
                        if elapsed >= t and count < t:
                            print(f'{t}...')
                            count += 1

                print('Cheese!')
                Image.fromarray(frame).save(f'assets/test/eCub{i}_rgb.png')
                Image.fromarray(depth).save(f'assets/test/eCub{i}_depth.png')
                time.sleep(1)
                print(f'Images saved as assets/test/eCub{i}_rgb/depth.png')
                i += 1
            elif key == ord('q'):
                exit()