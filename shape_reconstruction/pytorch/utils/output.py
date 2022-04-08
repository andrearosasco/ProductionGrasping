import time
import random
import open3d as o3d
import numpy as np

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries, Visualizer
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector, Vector2iVector
    from open3d.cpu.pybind.visualization import draw_geometries, Visualizer
    from open3d.cpu.pybind.geometry import PointCloud, LineSet


class PoseVisualizer:
    def __init__(self, device="cuda", live=False):

        # Random seed
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

        # Class parameters
        self.device = device
        self.live = live

        # TODO UPDATE OPEN3D
        # ctr = self.vis.get_view_control()
        # parameters = o3d.io.read_pinhole_camera_parameters("assets/gazebo_camera_reference.json")
        # ctr.convert_from_pinhole_camera_parameters(parameters, True)

        # Complete point cloud
        self.complete_pc = PointCloud()
        self.complete_pc.points = Vector3dVector(np.random.randn(2348, 3))
        # Partial point cloud
        self.partial_pc = PointCloud()
        self.partial_pc.points = Vector3dVector(np.random.randn(2024, 3))
        # Coords
        self.coords_mesh = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) for _ in range(2)]
        # Room
        self.line = LineSet()
        # Add if live, create visualizer and add geometries
        if self.live:
            self.vis = Visualizer()
            self.vis.create_window(width=1920, height=1080)
            self.vis.add_geometry(self.complete_pc)
            self.vis.add_geometry(self.partial_pc)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            for mesh in self.coords_mesh:
                self.vis.add_geometry(mesh)
            self.vis.add_geometry(self.line)

    def run(self, complete_pc_aux, poses, mean=(0, 0, 0), var=0.5, depth_pc=None):
        """
        It visualizes the results, given all the elements
        Args:
            partial_pc_aux: PointCloud, the partial one
            complete_pc_aux: PointCloud, the complete one
            poses: Tuple(np.array(3), np.array(3), np.array(3), np.array(3)), which are respectively first best center,
                first best normal, second best center, second best normal
            mean: np.array(3), the mean of the real partial point cloud
            var: Int, the variance of the real partial point cloud
            depth_pc: Point Cloud, the point cloud directly reconstructed from the depth image
        Returns: None
        """
        geometries = []

        best_centers = (poses[0], poses[2])
        best_rots = (poses[1], poses[3])

        # Orient poses
        i = 0
        for c, R, coord_mesh in zip(best_centers, best_rots, self.coords_mesh):
            coord_mesh_ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_mesh_.rotate(R, center=[0, 0, 0])
            coord_mesh_.translate(c, relative=False)

            # Update mesh
            coord_mesh.triangles = coord_mesh_.triangles
            coord_mesh.vertices = coord_mesh_.vertices

            if self.live:
                self.vis.update_geometry(coord_mesh)
            else:
                geometries.append(coord_mesh)

            i += 1

        # Update partial point cloud in visualizer
        # NOTE: this point cloud (after the de normalization) overlaps with the point cloud reconstructed from the depth
        # self.partial_pc.clear()
        # self.partial_pc += partial_pc_aux
        # colors = np.array([0, 255, 0])[None, ...].repeat(len(self.partial_pc.points), axis=0)
        # self.partial_pc.colors = Vector3dVector(colors)
        # # Invert x axis to plot like the pc obtained from depth
        # inverted = np.array(self.partial_pc.points)
        # # inverted[..., 2] = -inverted[..., 2]
        # self.partial_pc.points = Vector3dVector(inverted)
        # # De-normalize
        # self.partial_pc.scale(var * 2, center=[0, 0, 0])
        # self.partial_pc.translate(mean)

        # Update complete point cloud in visualizer
        self.complete_pc.clear()
        self.complete_pc += complete_pc_aux
        colors = np.array([255, 0, 0])[None, ...].repeat(len(self.complete_pc.points), axis=0)
        self.complete_pc.colors = Vector3dVector(colors)
        # Invert x axis to plot like the pc obtained from the depth
        inverted = np.array(self.complete_pc.points)
        # inverted[..., 2] = -inverted[..., 2]
        self.complete_pc.points = Vector3dVector(inverted)
        # De-normalize
        self.complete_pc.scale(var * 2, center=[0, 0, 0])
        self.complete_pc.translate(mean)

        # Update visualizer
        if self.live:
            # self.vis.update_geometry(self.partial_pc)
            self.vis.update_geometry(self.complete_pc)
            self.vis.poll_events()
            self.vis.update_renderer()
        else:
            o3d.visualization.draw_geometries([self.complete_pc] + geometries)
