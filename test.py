import cv2
from apriltag import Apriltag
import tagUtils as tud
import numpy as np

ap = Apriltag()
ap.create_detector(debug=True)
filename = 'tag.png'
frame = cv2.imread(filename)
detections = ap.detect(frame)
if len(detections) > 0:
    print('识别成功')
else:
    print('识别失败')
show = frame
edges = np.array([[0, 1],
                [1, 2],
                [2, 3],
                [3, 0]])
for detection in detections:
    point = tud.get_pose_point(detection.homography)
    #dis = tud.get_distance(detection.homography,122274)
    for j in range(4):
        cv2.line(show,tuple(point[edges[j,0]]),tuple(point[edges[j,1]]),(0,0,255),2)
    cv2.imwrite("res.png",show)
    # cv2.imshow("window", show)
    # cv2.waitKey(0)