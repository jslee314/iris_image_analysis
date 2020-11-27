# import cv2
# import glob
#
# images_path = "D:/2. data/total_iris/original_631/"
#
# images = sorted(glob.glob(images_path + "*.png"))
#
# for src in images:
#     img = cv2.imread(src, 0)
#     img = cv2.resize(img, (500, 500))
#     _, thr = cv2.threshold(img, 15, 250, 1, cv2.THRESH_BINARY)
#     cv2.imshow("img", thr);
#
#     eq = cv2.equalizeHist(img)
#     _, thr2 = cv2.threshold(eq, 4, 250, 1,cv2.THRESH_BINARY)
#     cv2.imshow("eq", thr2);
#
#     cv2.waitKey()
#
#
#
#
#
#
#  #####################################smoothing image Test
#
# import cv2
# import numpy as np
#
# def nothing(x):
#     pass
#
# src = "D:/2. data/entropy/1.png"
# img = cv2.imread(src, 0)
# img = cv2.resize(img, (500, 500))
#
# cv2.namedWindow('image')
# cv2.createTrackbar('K', 'image', 1, 250, nothing)
#
# while(1):
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#     k = cv2.getTrackbarPos('K', 'image')
#
#     #(0,0)이면 에러가 발생함으로 1로 치환
#     if k == 0:
#         k = 1
#
#     # trackbar에 의해서 (1,1) ~ (20,20) kernel생성
#     kernel = np.ones((k, k), np.float32)/(k*2)
#     dst = cv2.filter2D(img, -1, kernel)
#
#     cv2.imshow('image', dst)
#
# cv2.destroyAllWindows()