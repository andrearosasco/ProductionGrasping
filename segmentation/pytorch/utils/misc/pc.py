import numpy as np
from open3d.cpu.pybind.pipelines.registration import registration_icp


class ICP:
    def __init__(self, model, threshold=1, trans_init=np.eye(4)):
        self.call = 0
        self.model = model
        self.threshold = threshold
        self.trans_init = trans_init

    def __call__(self, target):
        # if len(target.points) != 0:
        #     t = np.eye(4)
        #     t[:3, 3] = np.array(target.points)[0] - self.model.get_position()
        #     self.model.transform(t)

        # if self.call < 5:
        #     source = hidden_remove(self.model.pc)
        # else:
        #     source = self.model.pc
        # self.call += 1
        source = self.model.pc

        reg_p2p = registration_icp(
            source,
            target,
            self.threshold,
            init=self.trans_init,
            # estimation_method=TransformationEstimationPointToPoint(),
            # criteria=ICPConvergenceCriteria(relative_fitness=1e-16, relative_rmse=1e-16, max_iteration=100000)
        )
        self.model.transform(reg_p2p.transformation)

        return reg_p2p