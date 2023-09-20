import cv2
import numpy as np
import math
import os

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

if __name__ == "__main__":
    fold = "./TY0913"
    file_list = ["-10.png","-5.png","0.png","1.png","2.png","3.png","4.png",
                 "5.png","6.png","10.png"]
    list1 = []
    list2 = []
    for img_file in file_list:
        # img_file = "-5.png"
        file_name = img_file.split('.')[0]
        img_path = os.path.join(fold,img_file)
        coord_path = os.path.join(fold,"coord_"+file_name+".txt")
        #print("file:{}".format(img_file))

        tagsize = 0.0625
        fx = 591.797  
        fy = 591.829  
        cx = 373.161
        cy = 203.246
        half_tagsize = tagsize / 2.0

        Kmat = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0,  1 ]],dtype=np.float64)
        disCoeffs= np.zeros([4, 1],dtype=np.float64) * 1.0

        opoints = np.array([[-1.0, -1.0, 0.0],
                            [1.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [-1.0, 1.0, 0.0],
                            [0, 0, 0.0]],dtype=np.float64) * half_tagsize
        campoint = []
        content = open(coord_path)
        for line in content:
            x,y = line.split(',')
            x = x.strip()
            y = y.strip()
            x = float(x)
            y = float(y)
            campoint.append([x,y])
        campoint = np.array(campoint,dtype=np.float64)

        rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs)
        #print("rvec={}".format(rvec))
        #print("tvec={}".format(tvec))
        rotate_m,_ = cv2.Rodrigues(rvec) # world to cam; x_cm = R * x_w + t; x_pixel = Kmat * x_cm
        #print("R={}".format(rotate_m))
        yaw,pitch,roll = rotationMatrixToEulerAngles(rotate_m)
        # print("相机相对标签的欧拉角 yaw {} pitch {} roll {}".format(yaw,pitch,roll))
        # print("相机相对标签")
        # print("file {} rx {} ry {} rz {} tx {} ty {} tz {}".
        #       format(img_file,roll,pitch,yaw,tvec[0],tvec[1],tvec[2]))
        list1.append("file {} rx {} ry {} rz {} tx {} ty {} tz {}".
              format(img_file,roll,pitch,yaw,tvec[0],tvec[1],tvec[2]))

        inv = np.linalg.inv(rotate_m)

        R_c = inv # 相机坐标系到世界坐标系的转换
        T_c = -R_c @ tvec
        rvec_t = cv2.Rodrigues(R_c)[0]
        tvec_t = T_c
        #print("rvec_t={}".format(rvec_t))
        #print("tvec_t={}".format(tvec_t))
        yaw,pitch,roll = rotationMatrixToEulerAngles(R_c)
        #print("标签相对相机的欧拉角 yaw {} pitch {} roll {}".format(yaw,pitch,roll))

        dis  =math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
        #print("distance={} m".format(dis))
        
        frame = cv2.imread(img_path)

        center = np.array([[0, 0, 0.0],[0, 0, -2],[1,0,0],[0,1,0]],dtype=np.float64) * half_tagsize

        point, jac = cv2.projectPoints(center, rvec, tvec, Kmat, disCoeffs)
        point = np.int32(np.reshape(point,[4,2]))
        # cv2.line(frame,tuple(point[0]),tuple(point[1]),(0,0,255),2)
        # cv2.line(frame,tuple(point[0]),tuple(point[2]),(0,255,0),2)
        # cv2.line(frame,tuple(point[0]),tuple(point[3]),(255,0,0),2)
        # cv2.imwrite(os.path.join(fold,file_name+"_res.png"),frame)
        # print("标签相对相机")
        # print("file {} rx {} ry {} rz {} tx {} ty {} tz {}".
        #       format(img_file,roll,pitch,yaw,tvec_t[0],tvec_t[1],tvec_t[2]))
        list2.append("file {} rx {} ry {} rz {} tx {} ty {} tz {}".
               format(img_file,roll,pitch,yaw,tvec_t[0],tvec_t[1],tvec_t[2]))

print("相机相对标签(旋转矩阵和平移矩阵是将标签坐标转换到相机坐标)")
for a in list1:
    print(a)
print("标签相对相机(旋转矩阵和平移矩阵是将相机系下的坐标转换到标签坐标)")
for a in list2:
    print(a)