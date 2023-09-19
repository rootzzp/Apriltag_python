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
                     [0, 0, 1]])
disCoeffs= np.zeros([4, 1]) * 1.0

opoints = np.array([[-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [0, 0, 0.0]]) * half_tagsize

campoint = np.array([[314, 93,  0.0],
                     [422, 79,  0.0],
                     [449, 186, 0.0],
                     [338, 207, 0.0],
                     [381, 143, 0.0]])

rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs)
filename = './TY0913/-5.png'
frame = cv2.imread(filename)

point, jac = cv2.projectPoints(opoints, rvec, tvec, Kmat, disCoeffs)
point = np.int32(np.reshape(point,[4,2]))
cv2.line(frame,point[4],point[0],(255,0,0))
cv2.line(frame,point[4],point[1],(0,255,0))
cv2.line(frame,point[4],point[2],(0,0,255))
cv2.imshow("show",frame)
cv2.waitKey(0)
cv2.imwrite("dst.png",frame)