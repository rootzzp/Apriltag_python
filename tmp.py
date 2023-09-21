from spatialmath import SO3, SE3
from spatialmath.base import *
import math
R1 = SO3.Rx(0.3)
R2 = SO3.Rz(30, 'deg')
T = SE3(1,2,3) * SE3.Rx(30, 'deg')
T.printline()
T.plot()
trplot( transl(1,2,3), frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10])
trplot( transl(3,1, 2), color='red', width=3, frame='B')
trplot( transl(4, 3, 1)@trotx(math.pi/3), color='green', frame='c', dims=[0,4,0,4,0,4])

# tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200)
# tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200, movie='out.mp4')

import numpy as np
import cv2
R_10 = np.array([[ 0.92425649, -0.19320705,  0.32927342], # 当前
 [ 0.23383767,  0.96826449, -0.08822594],
 [-0.30177788,  0.15853992,  0.94010382]])

R_30 = np.array([[ 0.93594737, -0.11260858, -0.33364925], # 参考
 [ 0.04449818,  0.97771647, -0.20515949],
 [ 0.34931708,  0.1771717,   0.92010204]])

# R = R_10 @ R_30
R = np.transpose(R_10) @ np.transpose(R_30)
rvec,_ = cv2.Rodrigues(R)
tmp = math.sqrt(rvec[0]**2+rvec[1]**2+rvec[2]**2)
angle = math.degrees(tmp)
print("-10.png 相对 30.png angle = {:.2f} degree".format(angle))