import copy

import numpy as np

from ransac.trt_ransac import TrTRansac
from shape_reconstruction.pytorch.utils.misc import project_onto_plane, angle_between
from shape_reconstruction.pytorch.utils.pose_generator import PoseGenerator


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y


class GraspEstimator:
    def __init__(self):
        self.ransac = TrTRansac()
        pass

    @staticmethod
    def create_rotation_matrix(a, b):
        """
        Creates the rotation matrix such that b = R @ a
        from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        Args:
            a: np.array (,3), unit vector already normalized
            b: np.array (,3), unit vector already normalized
        Returns:
            R: np.array(3, 3), transformation matrix
        """
        v = np.cross(a, b)

        c = np.dot(a, b)

        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        R = np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))
        return R

    def find_poses(self, points, dist, iterations):
        """
        Get a complete point cloud and return the a point cloud with good grasping spot
        Args:
            up:
            var:
            mean:
            debug: if to draw each iteration with open3d.draw_geometries
            dist: distance of points in RANSAC algorithm to belong to plane
            iterations: number of iterations to do in segment plane
            n_points: number of points to use in each iteration of segment plane
            pc: Complete Point Cloud
        Returns:
            poses: first best center, first best normal, second best center, second best normal
        """
        var = 0.5
        mean = np.array([0, 0, 0])
        up = False
        align_front = False

        candidates = self.ransac(points, dist, iterations)
        if candidates is None:
            return None

        planes, points = candidates

        normals = planes[..., :3]
        centers = np.array([np.mean(surface, axis=0) for surface in points])

        right = np.argmax(centers[:, 0], axis=0)

        right_normal = normals[right]
        right_center = centers[right]

        angle = (normals @ right_normal)
        angle[right] = 0
        left = np.argmax(np.abs(angle))
        left_plane = planes[left]
        left_normal = normals[left]

        p0 = np.array([0, 0, -left_plane[3] / left_plane[2]])
        d = ((p0 - right_center) @ left_normal) / (right_normal @ left_normal)
        left_center = right_center + right_normal * d

        aux = copy.deepcopy(centers)
        aux[[right, left]] = np.array([-1, -1, -1]) # Because 0 can still be greater than every other elements
        front = np.argmax(aux[:, 2], axis=0)
        front_normal = normals[front]

        if right_normal[0] < 0:
            right_normal = -right_normal
        right_normal = -right_normal
        if front_normal[2] > 0:
            front_normal = -front_normal

        # Makes the z axis parallel to the normal of the plane
        rotation = origin2pose([{'from': np.array([0, 0, 1]), 'to': right_normal},
                     {'from': np.array([1, 0, 0]), 'to': front_normal}]) # , right_center, left_center, points, viz

        return [right_center, rotation, left_center, copy.deepcopy(rotation), planes, points]


def origin2pose(constraints): # , c1, c2, points, viz
    """Return the transformation from the origin frame to the
     hand pose satisfying the given constraints."""

    R1 = PoseGenerator.create_rotation_matrix(np.array(constraints[0]['from']), constraints[0]['to'])

    y = (R1 @ constraints[1]['from']) / np.linalg.norm(R1 @ constraints[1]['from'])
    projected = project_onto_plane(constraints[1]['to'], constraints[0]['to'])  # TODO CAREFull TO POINT DOWN
    projected = np.array(projected) / np.linalg.norm(projected)

    # Compute angle between projected y axe and actual y axe
    ort = np.cross(y, projected) / np.linalg.norm(np.cross(y, projected))
    sign = round(ort @ constraints[0]['to']) # this either get rounded to 1 or -1

    rotation_radians = angle_between(y, projected) * sign
    # print(np.degrees(rotation_radians))  # TODO remove debug

    # Rotate mesh
    C = np.array([[0, -constraints[0]['to'][2], constraints[0]['to'][1]],
                  [constraints[0]['to'][2], 0, -constraints[0]['to'][0]],
                  [-constraints[0]['to'][1], constraints[0]['to'][0], 0]])
    R2 = np.eye(3) + C * np.sin(rotation_radians) + C @ C * (1 - np.cos(rotation_radians))

    # viz(points, R1, R2, c1, c2)

    return R2 @ R1

