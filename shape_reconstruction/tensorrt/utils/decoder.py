import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from shape_reconstruction.pytorch.configs import TrainConfig, ModelConfig
from shape_reconstruction.pytorch.model import PCRNetwork


class Decoder:
    def __init__(self):

        model = PCRNetwork.load_from_checkpoint('./shape_reconstruction/pytorch/checkpoint/final', config=ModelConfig)

        model.eval()

        self.sdf = model.sdf

    def __call__(self, fast_weights):

        refined_pred = torch.tensor(torch.randn(1, 10000, 3).cpu().detach().numpy() * 1, device=TrainConfig.device,
                                    requires_grad=True)

        loss_function = BCEWithLogitsLoss(reduction='mean')
        optim = Adam([refined_pred], lr=0.1)

        c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
        new_points = [] # refined_pred.detach().clone()
        for step in range(20):
            results = self.sdf(refined_pred, fast_weights)
            new_points += [refined_pred.detach().clone()[:, (torch.sigmoid(results).squeeze() >= 0.5) * (torch.sigmoid(results).squeeze() <= 1), :]]

            gt = torch.ones_like(results[..., 0], dtype=torch.float32)
            gt[:, :] = 1
            loss1 = c1 * loss_function(results[..., 0], gt)

            loss_value = loss1

            self.sdf.zero_grad()
            optim.zero_grad()
            loss_value.backward(inputs=[refined_pred])
            optim.step()

        ##################################################
        ################# Visualization ##################
        ##################################################
        selected = torch.cat(new_points, dim=1).cpu().squeeze().numpy()
        return selected

