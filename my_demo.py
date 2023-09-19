import cv2
import numpy as np

tagsize = 0.0625  
fx = 591.797  
fy = 591.829  
cx = 373.161
cy = 203.246
half_tagsize = tagsize / 2.0

Kmat = np.array([[591.797, 0, 373.161],
                     [0, 591.829, 203.246],
                     [0, 0, 1]],dtype=np.float64)
disCoeffs= np.zeros([4, 1],dtype=np.float64) * 1.0

opoints = np.array([[-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [0, 0, 0.0]],dtype=np.float64) * half_tagsize

campoint = np.array([[314, 93],
                     [422, 79],
                     [449, 186],
                     [338, 207],
                     [381, 143]],dtype=np.float64)

rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs)
print(rvec)
print(tvec)
filename = './TY0913/-5.png'
frame = cv2.imread(filename)

point, jac = cv2.projectPoints(opoints, rvec, tvec, Kmat, disCoeffs)
point = np.int32(np.reshape(point,[5,2]))
print(point[4])
a = tuple(point[0])
cv2.line(frame,tuple(point[4]),tuple(point[0]),(255,0,0),1)
cv2.line(frame,tuple(point[4]),tuple(point[1]),(0,255,0),1)
cv2.line(frame,tuple(point[4]),tuple(point[2]),(0,0,255),1)
# cv2.imshow("show",frame)
# cv2.waitKey(0)
cv2.imwrite("dst.png",frame)