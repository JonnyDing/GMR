# Python: 读取 JSON 字典 -> 前向运动学 -> matplotlib 3D 可视化

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# 示例 JSON（请替换为你自己的完整数据结构）
json_data = {
    "pelvis": ["Hips", 0, 10, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
    "left_hip_yaw_link": ["LeftUpLeg", 0, 10, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, -0.5]],
    "left_knee_link": ["LeftLeg", 0, 10, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, -0.5]],
    "left_ankle_roll_link": ["LeftFootMod", 50, 10, [0.0, 0.0, 0.0], [-0.70710678, 0.70710678, 0.0, 0.0]],
    "right_hip_yaw_link": ["RightUpLeg", 0, 10, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, -0.5]],
    "right_knee_link": ["RightLeg", 0, 10, [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, -0.5]],
    "right_ankle_roll_link": ["RightFootMod", 50, 10, [0.0, 0.0, 0.0], [-0.70710678, 0.70710678, 0.0, 0.0]],
    "torso_link": ["Spine2", 0, 100, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
    "left_shoulder_yaw_link": ["LeftArm", 0, 100, [0.0, 0.0, 0.0], [0.5, 0.5, -0.5, -0.5]],
    "left_elbow_link": ["LeftForeArm", 0, 10, [0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]],
    "left_wrist_yaw_link": ["LeftHand", 0, 10, [0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]],
    "right_shoulder_yaw_link": ["RightArm", 0, 100, [0.0, 0.0, 0.0], [0.5, 0.5, -0.5, -0.5]],
    "right_elbow_link": ["RightForeArm", 0, 10, [0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]],
    "right_wrist_yaw_link": ["RightHand", 0, 10, [0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]]
}

# 预设父子关系（请根据实际数据调整）
parent_map = {
    "pelvis": None,
    "torso_link": "pelvis",
    "left_hip_yaw_link": "pelvis",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_roll_link": "left_knee_link",
    "right_hip_yaw_link": "pelvis",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_roll_link": "right_knee_link",
    "left_shoulder_yaw_link": "torso_link",
    "left_elbow_link": "left_shoulder_yaw_link",
    "left_wrist_yaw_link": "left_elbow_link",
    "right_shoulder_yaw_link": "torso_link",
    "right_elbow_link": "right_shoulder_yaw_link",
    "right_wrist_yaw_link": "right_elbow_link"
}

def quat_to_rot_matrix(q):
    # q = [w,x,y,z]
    w, x, y, z = q
    n = sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])
    return R

# 计算全局变换
global_tf = {}
def compute_global_transforms():
    pending = set(json_data.keys())
    while pending:
        progressed = False
        for node in list(pending):
            parent = parent_map.get(node, None)
            entry = json_data.get(node)
            if entry is None:
                pending.remove(node)
                progressed = True
                continue
            local_pos = np.array(entry[3], dtype=float)
            local_q = entry[4]
            local_R = quat_to_rot_matrix(local_q)
            if parent is None:
                global_tf[node] = (local_R, local_pos)
                pending.remove(node)
                progressed = True
            else:
                if parent in global_tf:
                    R_p, t_p = global_tf[parent]
                    R_g = R_p @ local_R
                    t_g = R_p @ local_pos + t_p
                    global_tf[node] = (R_g, t_g)
                    pending.remove(node)
                    progressed = True
        if not progressed:
            break

compute_global_transforms()

# 提取位置并绘图
joint_positions = {n: tf[1] for n, tf in global_tf.items()}
for n in json_data.keys():
    if n not in joint_positions:
        joint_positions[n] = np.array(json_data[n][3], dtype=float)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Skeleton Visualization (forward kinematics)")
xs = [p[0] for p in joint_positions.values()]
ys = [p[1] for p in joint_positions.values()]
zs = [p[2] for p in joint_positions.values()]
ax.scatter(xs, ys, zs)
for node, parent in parent_map.items():
    if parent is None: continue
    p_child = joint_positions.get(node)
    p_parent = joint_positions.get(parent)
    if p_child is None or p_parent is None:
        continue
    ax.plot([p_parent[0], p_child[0]], [p_parent[1], p_child[1]], [p_parent[2], p_child[2]])
# 统一坐标轴尺度（便于观看）
max_range = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1e-6)
mid_x = (max(xs)+min(xs))/2
mid_y = (max(ys)+min(ys))/2
mid_z = (max(zs)+min(zs))/2
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
plt.show()
