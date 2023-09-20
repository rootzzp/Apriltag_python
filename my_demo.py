import cv2
import numpy as np
import math

def isRotationMatrix(R) :
    # 得到该矩阵的转置
    Rt = np.transpose(R)
    # 旋转矩阵的一个性质是，相乘后为单位阵
    shouldBeIdentity = np.dot(Rt, R)
    # 构建一个三维单位阵
    I = np.identity(3, dtype = R.dtype)
    # 将单位阵和旋转矩阵相乘后的值做差
    n = np.linalg.norm(I - shouldBeIdentity)
    # 如果小于一个极小值，则表示该矩阵为旋转矩阵
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    # 断言判断是否为有效的旋转矩阵
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([math.degrees(z), math.degrees(y), math.degrees(x)])

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

campoint = np.array([[314, 93 ],
                     [422, 79 ],
                     [449, 186],
                     [338, 207],
                     [381, 143]],dtype=np.float64)

rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs)
print("rvec",rvec)
print("tvec",tvec)
rotate_m,_ = cv2.Rodrigues(rvec)
print("R={}".format(rotate_m))
yaw,pitch,roll = rotationMatrixToEulerAngles(rotate_m)
print("yaw {} pitch {} roll {}".format(yaw,pitch,roll))

dis  =math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
print("distance={}".format(dis))
filename = './TY0913/-5.png'
frame = cv2.imread(filename)

center = np.array([[0, 0, 0.0],[0, 0, -2],[1,0,0],[0,1,0]],dtype=np.float64) * half_tagsize

point, jac = cv2.projectPoints(center, rvec, tvec, Kmat, disCoeffs)
point = np.int32(np.reshape(point,[4,2]))
cv2.line(frame,tuple(point[0]),tuple(point[1]),(0,0,255),2)
cv2.line(frame,tuple(point[0]),tuple(point[2]),(0,255,0),2)
cv2.line(frame,tuple(point[0]),tuple(point[3]),(255,0,0),2)
cv2.imwrite("dst.png",frame)