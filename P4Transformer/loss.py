import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.body_model.lbs import lbs
import matplotlib.pyplot as plt
import cv2
import os
import json
from utils import rodrigues_2_rot_mat

SMPLX_MODEL_NEUTRAL_PATH = 'path_to_neutral_model'
SMPLX_MODEL_FEMALE_PATH = 'path_to_female_model'
SMPLX_MODEL_MALE_PATH = 'path_to_male_model'

class LossManager():
    def __init__(self, ding_bot=None) -> None:
        super(LossManager).__init__()
        self.loss_dict = {}
        self.batch_loss = []
        self.ding_bot = ding_bot

    def update_loss(self, name, loss):
        if name not in self.loss_dict:
            self.loss_dict.update({name:[loss]})
        else:
            self.loss_dict[name].append(loss)

    def calculate_total_loss(self):
        batch_loss = []
        for loss in self.loss_dict.values():
            batch_loss.append(loss[-1])
        total_loss = torch.sum(torch.stack(batch_loss))
        self.batch_loss.append(total_loss)
        return total_loss

    def calculate_epoch_loss(self, output_path, epoch):
        fig = plt.figure()
        loss_json = os.path.join(output_path, "loss.json")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            open(loss_json, "w")
        with open(loss_json, "r") as f:
            losses = json.load(f) if epoch else dict()
        with open(loss_json, "w") as f:
            losses.update({"epoch":epoch})
            for i, (name, loss) in enumerate(self.loss_dict.items()):
                epoch_loss = np.hstack((losses.get(name, []), np.average(torch.tensor(loss))))
                losses.update({name:list(epoch_loss)})
                fig.add_subplot(3, 3, i+1, title=name).plot(epoch_loss)
            total_loss = np.hstack((losses.get("total_loss", []), np.average(torch.tensor(self.batch_loss))))
            losses.update({"total_loss":list(total_loss)})
            json.dump(losses, f)
        fig.add_subplot(3, 3, i+2, title="total_loss").plot(total_loss)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close()
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        if self.ding_bot:
            self.ding_bot.add_md("train_mmbody", "【IMG】 \n ![img]({}) \n 【LOSS】\n epoch={}, loss={}".format(self.ding_bot.img2b64(img), epoch, total_loss[-1]))
            self.ding_bot.enable()
        
        self.loss_dict = {}
        self.batch_loss = []

    def calculate_test_loss(self, output_path):
        fig = plt.figure()
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        loss_json = os.path.join(output_path, "loss.json")
        losses = {}
        for i, (name, loss) in enumerate(self.loss_dict.items()):
            _loss = np.sort(torch.tensor(loss))
            losses.update({name:np.mean(_loss).tolist()})
            hist, bin_edges = np.histogram(_loss, bins=100)
            cdf = np.cumsum(hist)/len(_loss)
            fig.add_subplot(2, 3, i+1, title=name).plot(bin_edges[:-1], cdf)
        total_loss = np.average(torch.tensor(self.batch_loss))
        losses.update({"total_loss":total_loss.tolist()})
        with open(loss_json, "w") as f:
            json.dump(losses, f)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close()
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        np.save(os.path.join(output_path, "joints_loss"), np.sort(torch.tensor(self.loss_dict["joints_loss"])))
        np.save(os.path.join(output_path, "vertices_loss"), np.sort(torch.tensor(self.loss_dict["vertices_loss"])))

class GeodesicLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(GeodesicLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            # breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta


class SMPLXModel(BodyModel):
    def __init__(self, num_betas=16, **kwargs):
        super().__init__(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=num_betas, num_expressions=0, **kwargs)

    def forward(self, pose_body, betas, use_rodrigues=True):
        device = pose_body.device
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression', 
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

        batch_size = pose_body.shape[0]
        trans = pose_body[:, :3]
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1)
        if not use_rodrigues:
            init_pose = rodrigues_2_rot_mat(init_pose)
        full_pose = torch.cat([pose_body[:, 3:], init_pose], dim=-1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)

        joints = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        return dict(vertices=verts, joints=joints)
    
class MeshLoss(_Loss):
    smplx_model = [None, None, None]
    def __init__(self, device: torch.device = torch.device('cpu'), size_average=None, reduce=None, reduction: str = 'mean', scale: float = 1) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        if self.smplx_model[0] is None:
            self.smplx_model[0] = SMPLXModel(bm_fname=SMPLX_MODEL_FEMALE_PATH, num_betas=16, num_expressions=0, device=device)
        if self.smplx_model[1] is None:
            self.smplx_model[1] = SMPLXModel(bm_fname=SMPLX_MODEL_MALE_PATH, num_betas=16, num_expressions=0, device=device)
        if self.smplx_model[2] is None:
            self.smplx_model[2] = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, device=device)
        self.scale = scale

    def forward(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0, train: bool = True, use_rodrigues=False) -> torch.Tensor:

        _input = input * self.scale
        _target = target * self.scale

        if not use_gender:
            input_model = target_model = self.smplx_model[2]
        else:
            input_model = target_model = self.smplx_model[0 if target[0][-1] < 0.5 else 1]

        input_result = input_model(pose_body=_input[:,:-16], betas=_input[:,-16:], use_rodrigues=use_rodrigues)
        input_verts = input_result['verts']
        input_joints = input_result['joints']

        target_result = target_model(pose_body=_target[:,:-16], betas=_target[:,-16:], use_rodrigues=use_rodrigues)
        target_verts = target_result['verts']
        target_joints = target_result['joints']
        
        per_joint_err = torch.norm((input_joints - target_joints), dim=-1)
        per_vertex_err = torch.norm((input_verts - target_verts), dim=-1)

        if train:
            return (F.l1_loss(input_verts, target_verts, reduction=self.reduction), 
                    F.l1_loss(input_joints, target_joints, reduction=self.reduction))
        else:
            return (torch.sqrt(F.mse_loss(input_verts, target_verts, reduction=self.reduction)),
                    torch.sqrt(F.mse_loss(input_joints, target_joints, reduction=self.reduction)),
                    (per_joint_err, per_vertex_err))
