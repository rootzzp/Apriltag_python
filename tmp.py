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
tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200, movie='out.mp4')