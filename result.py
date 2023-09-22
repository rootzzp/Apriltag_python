import cv2
import numpy as np
import math
import os
import copy

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

def calc_relative_angle(T1,T2):
    T = np.linalg.inv(T1) @ T2
    # print(T)
    # R = np.transpose(R_10) @ np.transpose(R_30)
    R = T[:3,:3]
    rvec,_ = cv2.Rodrigues(R)
    tmp = math.sqrt(rvec[0]**2+rvec[1]**2+rvec[2]**2)
    angle = math.degrees(tmp)
    t = T[:3,3]
    # print(tmp)
    # theta = np.arccos((np.trace(R) - 1) / 2)
    # print(theta)
    return angle,R,t

if __name__ == "__main__":
    fold = "./TY0913"
    res_fold = "./result"
    file_list = ["-10.png","-5.png","0.png","1.png","2.png","3.png","4.png",
                 "5.png","6.png","10.png", "15.png", "20.png", "25.png", "30.png",]
    list1 = []
    list2 = []
    T_list = []
    for img_file in file_list:
        file_name = img_file.split('.')[0]
        img_path = os.path.join(fold,img_file)
        coord_path = os.path.join(fold,"coord_"+file_name+".txt")

        # tagsize = 0.0625
        tagsize = 62.5 # mm
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
                            ],dtype=np.float64) * half_tagsize
        campoint = []
        content = open(coord_path)
        for i,line in enumerate(content):
            if(i == 4):
                break
            x,y = line.split(',')
            x = x.strip()
            y = y.strip()
            x = float(x)
            y = float(y)
            campoint.append([x,y])
        campoint = np.array(campoint,dtype=np.float64)

        rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs,cv2.SOLVEPNP_DLS)
        rotate_m,_ = cv2.Rodrigues(rvec) # world to cam; x_cm = R * x_w + t; x_pixel = Kmat * x_cm
        yaw,pitch,roll = rotationMatrixToEulerAngles(rotate_m)
        # print(rotate_m)

        list1.append("tx {:.2f}mm, ty {:.2f}mm, tz {:.2f}mm".
              format(tvec[0][0],tvec[1][0],tvec[2][0]))

        inv = np.linalg.inv(rotate_m)

        R_c = inv # 相机坐标系到世界坐标系的转换
        T_c = -R_c @ tvec
        rvec_t = cv2.Rodrigues(R_c)[0]
        tvec_t = T_c
        yaw,pitch,roll = rotationMatrixToEulerAngles(R_c)
        T = np.zeros((4,4))
        T[3,3] = 1
        T[:3,:3] = R_c
        T[:3,3] = tvec.T
        T_list.append(copy.deepcopy(T))
        list2.append("file {} rx: {:.2f} degree, ry {:.2f} degree, rz {:.2f} degree, ".
               format(img_file,roll,pitch,yaw))

        dis  =math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
               
        frame = cv2.imread(img_path)
        img_h, img_w, _ = frame.shape
        img_center = (img_w // 2, img_h // 2)

        center = np.array([[0, 0, 0.0],[0, 0, 2],[1,0,0],[0,1,0]],dtype=np.float64) * half_tagsize

        point, jac = cv2.projectPoints(center, rvec, tvec, Kmat, disCoeffs)
        point = np.int32(np.reshape(point,[4,2]))
        cv2.line(frame,tuple(point[0]),tuple(point[1]),(0,0,255),2)
        cv2.putText(frame,'z',tuple(point[1]),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),1)

        cv2.line(frame,tuple(point[0]),tuple(point[2]),(0,255,0),2)
        cv2.putText(frame,'x',tuple(point[2]),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)

        cv2.line(frame,tuple(point[0]),tuple(point[3]),(255,0,0),2)
        cv2.putText(frame,'y',tuple(point[3]),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),1)

        cv2.line(frame,img_center,(img_w // 2 + 30, img_h // 2),(0,255,0),2)
        cv2.putText(frame,'x',(img_w // 2 + 30, img_h // 2),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)

        cv2.line(frame,img_center,(img_w // 2, img_h // 2 + 30),(255,0,0),2)
        cv2.putText(frame,'y',(img_w // 2, img_h // 2 + 30),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),1)

        # cv2.imwrite(os.path.join(res_fold,file_name+"_res.png"),frame)

        

for a,b in zip(list2,list1):
    print(a,b)
size = len(T_list)
for i in range(size):
    for j in range(size):
        angle,_,_ = calc_relative_angle(T_list[i],T_list[j])
        print("{} 相对 {} angle = {:.2f} degree".format(file_list[i],file_list[j],angle))

for i in range(size):
    for j in range(size):
        _,R,t = calc_relative_angle(T_list[i],T_list[j])
        yaw,pitch,roll = rotationMatrixToEulerAngles(R)
        print("{} 相对 {} rx: {:.2f} degree, ry {:.2f} degree, rz {:.2f} degree, tx {:.2f}mm, ty {:.2f}mm, tz {:.2f}mm".format(file_list[i],file_list[j],roll, pitch, yaw, t[0],t[1],t[2]))
