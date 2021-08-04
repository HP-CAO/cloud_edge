import numpy as np
from scipy.spatial.transform import Rotation as R

cam_p = np.array([0, 0, -0.5])
obj_p = np.array([0.1, 0.15, 0])

cam_r_euler = np.array([90, 0, 0])
obj_r_euler = np.array([0, 0, 0])

cam_w_r = R.from_euler('zxy', cam_r_euler, degrees=True).as_matrix()
obj_w_r = R.from_euler('zxy', obj_r_euler, degrees=True).as_matrix()

cam_w_t = cam_p.reshape(3, 1)
obj_w_t = obj_p.reshape(3, 1)
last = np.array([[0, 0, 0, 1]])

RT_w_c_inv = np.concatenate([cam_w_r.T, -1 * np.dot(cam_w_r.T, cam_w_t)], axis=-1)
RT_w_c_inv = np.concatenate([RT_w_c_inv, last], axis=0)

RT_w_ob = np.concatenate([obj_w_r, obj_w_t], axis=-1)
RT_w_ob = np.concatenate([RT_w_ob, last], axis=0)

RT_c_ob = np.dot(RT_w_c_inv, RT_w_ob)

obj_ori = np.array([0, 0, 0]).reshape(3, 1)
a = np.dot(RT_c_ob[:3, :3], obj_ori).squeeze()
b = RT_c_ob[:3, -1]
# obj_c_p = RT_c_ob[:3, :3] * obj_ori + RT_c_ob[:3, -1]
obj_c_p = a + b
print(a, b)
print(obj_c_p)

