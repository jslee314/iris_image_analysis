from find_radius.find_pupil_radius import get_radius, get_centor, undesired_objects
import cv2
import glob
import numpy as np

radius_path = "D:/2. data/radius/label/"
segmentations = sorted(glob.glob(radius_path+"*"))
input_height = 480
input_width = 640

for segName in segmentations:
    src = cv2.imread(segName, 1)
    src = cv2.resize(src, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    dst = src.copy()

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    largest_component_img = undesired_objects(gray)
    largest_component_img = largest_component_img.astype(np.uint8)

    cv2.imshow("largest_connectedComponent", largest_component_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2차원 분포의 1차 모멘트값 산정하기 -> 예측원의 중심 구하기
    cX, cY = get_centor(largest_component_img)

    # 도형의 contour의 좌표들과 도심간의 거리의 평균 -> 예측원의 반지름 구하기
    mean_radius = get_radius(largest_component_img, cX, cY)

    # 예측원과 중심 그리기
    cv2.circle(img=dst, center=(cX, cY), radius=3, color=(255, 0, 0), thickness=-1)
    cv2.circle(img=dst, center=(cX, cY), radius=mean_radius, color=(0, 0, 255), thickness=1)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
# radius_path = "D:/2. data/total_iris/iris_250/test/label/"
# segmentations = sorted(glob.glob(radius_path+"*"))
#
#
# for segName in segmentations:
#     src = cv2.imread(segName, 1)
#     gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
#     print(src)
#     print(gray)