# from image_momnet import image_statistics, image_statistics_2D
import glob
import cv2
import numpy as np
from skimage.filters.rank import entropy


src_path = "D:/2. data/total_iris/iris_pupil/pupil_200/"
image_paths = sorted(glob.glob(src_path + "*.png"))

images_rgb_np = []
images_bgr = []
image_names = []

print('start ====  ==== image_paths : %s', image_paths)

for i, image_path in enumerate(image_paths):
    print('%s\t%.1f' % (image_path, i))
    ############ 이미지 불러오기 ############
    image_name = image_path.split("\\")[-1]
    image_names.append(image_name)
    image_bgr = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    ############ 파일명에서 정보 추출하기 ############
    cX = int(image_path.split("_")[-4])
    cY = int(image_path.split("_")[-3])
    pupil_radius = int(image_path.split("_")[-2])
    iris_radius = int(image_path.split("_")[-1][:-4])
    segmentation_area = int(image_path.split("_")[-1][:-5])

    ############ 모멘트 분석 ############
    mask = np.where(image_bgr > 0, image_bgr, 0)
    hist = cv2.calcHist([image_bgr], [0], mask, [256], [0, 256])
    hist_image = np.zeros((image_bgr.shape[0], 256), dtype=np.uint8)
    for x, y in enumerate(hist):
        cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - y), 255)
    cv2.imshow("hist", hist_image)

    x = np.shape(hist)
    xp = np.sum(hist, axis=0)

    # 1차: centroid
    cx = np.sum(x * xp) / np.sum(xp)
    # 2차: standard deviation
    x2 = (x - cx) ** 2
    sx = np.sqrt(np.sum(x2 * xp) / np.sum(xp))
    # 3차: skewness
    x3 = (x - cx) ** 3
    skx = np.sum(xp * x3) / (np.sum(xp) * sx ** 3)

    # 4차: Kurtosis
    x4 = (x - cx) ** 4
    kx = np.sum(xp * x4) / (np.sum(xp) * sx ** 4)

    i1 = (cx, sx, skx, kx)
    names = ('Centroid', 'StdDev   ','Skewness', 'Kurtosis')
    for i, name in zip(i1, names):
        print('%s\t%.1f' %(name, i))


    # ############ entropy 분석 ############
    # from skimage.morphology import disk
    # import matplotlib.pyplot as plt
    # fig, (ax, ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=5, figsize=(18, 4))
    #
    # from PIL import Image
    # image_rgb = Image.open(image_path).convert("RGB")
    # ax.imshow(image_rgb)
    # ax.set_title("roi image")
    #
    # img0 = ax0.imshow(image_gray, cmap='gray')
    # ax0.set_title("gray")
    #
    # entr_img_3 = entropy(image_gray, disk(8))
    # entr_img_3 = np.where(entr_img_3 > 5.5, 255, 0)
    # normalized_entropy = np.sum(entr_img_3/255)/segmentation_area
    # entr_img_3 = ax1.imshow(entr_img_3)
    # ax1.set_title("Local 8(>5.5):  " + str(round(normalized_entropy)))
    #
    # entr_img_1 = entropy(image_gray, disk(5))
    # entr_img_1 = ax2.imshow(entr_img_1, cmap='viridis')
    # ax2.set_title("Local entropy 5")
    # ax2.axis("off")
    # fig.colorbar(entr_img_1, ax=ax2)
    #
    # entr_img_2 = entropy(image_gray, disk(8))
    # entr_img_2 = ax3.imshow(entr_img_2, cmap='viridis')
    # ax3.set_title("Local entropy 8")
    # ax3.axis("off")
    # fig.colorbar(entr_img_2, ax=ax3)
    #
    # fig.tight_layout()
    # # plt.show()
    # new_image_path = image_path[:-4]+"_plt.png"
    # plt.savefig(new_image_path)

    #
    # # # 예측원과 중심 그리기
    # # image_bgr = np.array(image_bgr)
    # # cv2.circle(img=image_bgr, center=(cX, cY), radius=3, color=(0, 0, 255), thickness=-1)
    # # cv2.circle(img=image_bgr, center=(cX, cY), radius=pupil_radius, color=(0, 0, 255), thickness=1)
    # # cv2.circle(img=image_bgr, center=(cX, cY), radius=new_pupil_radius, color=(0, 0, 255), thickness=1)
    # # cv2.circle(img=image_bgr, center=(cX, cY), radius=iris_radius, color=(0, 0, 255), thickness=1)
    # # cv2.circle(img=image_bgr, center=(cX, cY), radius=new_iris_radius, color=(0, 0, 255), thickness=1)
    # # cv2.imshow("image_bgr", image_bgr)
    #
    # # # ROI 그리기
    # # pupil_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_pupil_radius, color=(255, 255, 255), thickness=-1)
    # # iris_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_iris_radius, color=(255, 255, 255), thickness=-1)
    # # total_segmentation = np.where(pupil_segmentation == 255, 0, iris_seg)
    # # total_segmentation = np.where(iris_segmentation == 255, total_segmentation, 0)
    # # cv2.imshow("total_segmentation", total_segmentation)
    #
    # ## 30도씩 섹터별로 나누기
    # sectors_region = []
    # sectors_image = []
    # for i in range(12):
    #     sector = image_bgr.copy()
    #     startAngle = 270 + i * 30
    #     endAngle = startAngle + 30
    #     cv2.ellipse(img=sector, center=(cX, cY), axes=(iris_radius, iris_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(255, 0, 0), thickness=-1)
    #     cv2.ellipse(img=sector, center=(cX, cY), axes=(pupil_radius, pupil_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(0, 0, 0), thickness=-1)
    #     sectors_region.append(sector)
    #
    #     result_image = np.where(sector == (255, 0, 0), image_bgr, 0)
    #     sectors_image.append(sector)
    #
    #     # result = cv2.hconcat([sector, result_image])
    #     # cv2.imshow("result_"+str(i), result)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    #
    #     # # 방법 1: SIFT(Scale Invariant Feature Transform): 크기불변 이미지 특성 검출
    #     # gray_sector = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
    #     # sector2, sector3 = None, None
    #     #
    #     # sift = cv2.xfeatures2d.SIFT_create()
    #     # kp = sift.detect(gray_sector, None)
    #     #
    #     # sector2 = cv2.drawKeypoints(gray_sector, kp, sector2)
    #     # sector3 = cv2.drawKeypoints(gray_sector, kp, sector3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     # result = cv2.hconcat([sector2, sector3])
    #
    #     # 방법 2:
    #     gray_sector = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    #     hist = cv2.calcHist([gray_sector], [0], None, [256], [0, 256])
    #     hist_image = np.zeros((result_image.shape[0], 256), dtype=np.uint8)
    #
    #     for x, y in enumerate(hist):
    #         cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - y), 255)
    #
    #
    #
    #
    #     m = cv2.moments(gray_sector)
    #
    #     # caclulating the skewness
    #     skew = round(m['mu11'] / m['mu02'], 2)
    #
    #     # cv2.putText(gray_sector, str(skew), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    #     cv2.putText(gray_sector, str(round(m['m00']/100000)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    #     cv2.putText(gray_sector, str(round(m['m01']/10000000)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    #     cv2.putText(gray_sector, str(round(m['m02']/1000000000)), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    #     cv2.putText(gray_sector, str(round(m['m03']/100000000000)), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    #
    #     # cv2.putText(gray_sector, cv2.moments(hist_image)['m00'], (10, 10), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255,0), 2)
    #     # cv2.putText(gray_sector, cv2.moments(hist_image)['m00'], (10, 10), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255,0), 2)
    #
    #     dst = np.hstack([gray_sector, hist_image])
    #     #cv2.imshow("result_"+str(i), dst)
    #     cv2.imwrite("D:/2. data/total_iris/only_iris_4/"+str(i)+"_"+image_name, gray_sector)
    #     cv2.imshow("result_" + str(i), gray_sector)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #
