
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import torch_utils



qpos_left = torch.tensor([-0.45])

axis = torch.tensor([0.0,1.0,0.0])
ankle_quat = torch_utils.axis_angle_to_quat(axis,qpos_left)
print(ankle_quat) #tensor([[-0.0000, -0.2231, -0.0000,  0.9748]])
# 目标四元数 (无旋转)
target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])

# 计算从当前姿态到目标姿态的差值四元数
# delta_quat = target_quat * conjugate(current_quat)
current_conjugate = torch_utils.quat_conjugate(ankle_quat)
delta_quat = torch_utils.quat_mul(target_quat.unsqueeze(0), current_conjugate)
# 将差值四元数转换为轴角表示
delta_axis, delta_angle = torch_utils.quat_to_axis_angle(delta_quat)

print("修正旋转轴:", delta_axis)
print("修正旋转角度:", delta_angle)