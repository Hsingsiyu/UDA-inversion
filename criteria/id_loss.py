import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        # cos_target = torch.ones((n_samples, 1)).float().cuda()
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        id_logs = []
        loss = 0
        count = 0
        for i in range(n_samples):
            y_feat_detached = y_feats[i]
            diff_target=torch.mean((y_feat_detached-y_hat_feats[i])**2)
            # diff_input = y_hat_feats[i].dot(x_feats[i])
            # diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),})
            loss += diff_target
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        return loss / count, id_logs
