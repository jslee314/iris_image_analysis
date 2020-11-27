import cv2
import glob
import numpy as np
import math

def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    largest_connectedComponent = np.zeros(output.shape)
    largest_connectedComponent[output == max_label] = 255
    # cv2.imshow("Biggest component", largest_connectedComponent)
    # cv2.waitKey()
    return largest_connectedComponent


def get_centor(gray):
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        contour = np.array(contour).reshape(-1, 2)

        # 모멘트란,
        # 어떤 종류의 "물리적 효과"가 하나의 물리량 뿐만 아니라 그 물리량의 "분포상태"에 따라서 정해질 때, 정의되는 양
        # n차 모멘트 = (위치)^n(물리량)
        M = cv2.moments(contour)

        # 도심(무게중심)이란,
        # 단면의 직교좌표축에 대한 단면 1차 모멘트가 0이 되는 점이다.
        # 직교좌표축에서 도심까지의 거리를 구하는 방법은, 단면 1차 모멘트를 도형의 면적으로 나눈다.
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        return cX, cY



def get_radius(gray, cX, cY):
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    contour = np.array(contour).reshape(-1, 2)

    # 원의 반지름 : 도형의 contour의 좌표들과 도심간의 거리의 평균
    sum_radius = 0
    for x, y in contour:
        sum_radius = sum_radius + math.sqrt((x - cX) * (x - cX) + (y - cY) * (y - cY))

    mean_radius = round(sum_radius / len(contour))

    return mean_radius


radius_path = "D:/2. data/radius/image/"
segmentations = sorted(glob.glob(radius_path+"*"))
input_height = 320
input_width = 240

# for segName in segmentations:
#     src = cv2.imread(segName, 1)
#     src = cv2.resize(src, (input_height, input_width), interpolation=cv2.INTER_NEAREST)
#     dst = src.copy()
#     gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#     largest_connectedComponent = undesired_objects(gray)
#     cv2.imshow("largest_connectedComponent", largest_connectedComponent)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # 2차원 분포의 1차 모멘트값 산정하기 -> 예측원의 중심 구하기
#     cX, cY = get_centor(largest_connectedComponent)
#
#     # 도형의 contour의 좌표들과 도심간의 거리의 평균 -> 예측원의 반지름 구하기
#     mean_radius = get_radius(largest_connectedComponent, cX, cY)
#
#     # 예측원과 중심 그리기
#     cv2.circle(img=dst, center=(cX, cY), radius=3, color=(255, 0, 0), thickness=-1)
#     cv2.circle(img=dst, center=(cX, cY), radius=mean_radius, color=(0, 0, 255), thickness=1)
#
#     cv2.imshow("dst", dst)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


    #
    # dilation = cv2.dilate(im, kernel= np.ones((3, 3)), iterations=3)
    # erodeion = cv2.erode(dilation, kernel= np.ones((3, 3)), iterations=3)
    #
    # cv2.imshow("img", im)
    # cv2.imshow("dilation", dilation)
    # cv2.imshow("erodeion", erodeion)
# ########################### 방법 1
# for i, segName in enumerate(segmentations):
#     im = cv2.imread(segName, 1)
#
#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     ret, gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     contours, ii = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     contours = contours[0].reshape(-1, 2)
#     max = np.argmax(contours, axis=0)
#     min = np.argmin(contours, axis=0)
#
#     for c in contours[max]:
#         im = cv2.line(im, (c[0], c[1]), (c[0], c[1]), (0, 255,0), 4)
#
#     for c in contours[min]:
#         im = cv2.line(im, (c[0], c[1]), (c[0], c[1]), (0, 0,255), 4)
#
#     cv2.imshow("cimg", cimg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# ########################### 방법 2
# for i, segName in enumerate(segmentations):
#     img = cv2.imread(segName, 0)
#     # img = cv2.medianBlur(img, 5)
#     # img = cv2.Canny(img, 30, 70)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT,
#                                dp=1, minDist=10, param1=10, param2=10,
#                                minRadius=10, maxRadius=80)
#
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#
#         for i in circles[0, :]:
#             cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
#
#         cv2.imshow("cimg", cimg)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("원을 찾지 못했습니다.")