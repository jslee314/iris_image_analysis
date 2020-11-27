# from image_momnet import image_statistics, image_statistics_2D
import glob
import cv2
import numpy as np
from skimage.filters.rank import entropy

from skimage.morphology import disk
import matplotlib.pyplot as plt
from PIL import Image

src_path = "D:/2. data/total_iris/only_iris_40/"
image_paths = sorted(glob.glob(src_path + "*.png"))

images_rgb_np = []
images_bgr = []
image_names = []

for i, image_path in enumerate(image_paths):
    ############ 이미지 불러오기 ############
    image_name = image_path.split("\\")[-1]
    image_names.append(image_name)
    image_bgr = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = Image.open(image_path).convert("RGB")

    # led 비취는 부분 제거 한 rgb/gray image
    temp = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    image_rgb_noLed = np.where(temp > (160, 160, 160), 0, image_rgb)
    image_gray_noLed = np.where(image_gray > 160, 0, image_gray)

    ############ 파일명에서 정보 추출하기 ############
    cX = int(image_path.split("_")[-4])
    cY = int(image_path.split("_")[-3])
    pupil_radius = int(image_path.split("_")[-2])
    iris_radius = int(image_path.split("_")[-1][:-4])
    segmentation_area = int(image_path.split("_")[-1][:-5])

    ############ 모멘트 분석 ############
    # mask = np.where(image_bgr > 0, image_bgr, 0)
    # hist = cv2.calcHist([image_bgr], [0], mask, [256], [0, 256])
    # hist_image = np.zeros((image_bgr.shape[0], 256), dtype=np.uint8)
    # for x, y in enumerate(hist):
    #     cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - y), 255)
    # # cv2.imshow("hist", hist_image)
    #
    # x = np.shape(hist)
    # xp = np.sum(hist, axis=0)
    #
    # # 1차: centroid
    # cx = np.sum(x * xp) / np.sum(xp)
    # # 2차: standard deviation
    # x2 = (x - cx) ** 2
    # sx = np.sqrt(np.sum(x2 * xp) / np.sum(xp))
    # # 3차: skewness
    # x3 = (x - cx) ** 3
    # skx = np.sum(xp * x3) / (np.sum(xp) * sx ** 3)
    #
    # # 4차: Kurtosis
    # x4 = (x - cx) ** 4
    # kx = np.sum(xp * x4) / (np.sum(xp) * sx ** 4)
    #
    # i1 = (cx, sx, skx, kx)
    # names = ('Centroid', 'StdDev   ','Skewness', 'Kurtosis')
    # for i, name in zip(i1, names):
    #     print('%s\t%.1f' %(name, i))

    # # 예측원과 중심 그리기
    # image_bgr = np.array(image_bgr)
    # cv2.circle(img=image_bgr, center=(cX, cY), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(img=image_bgr, center=(cX, cY), radius=pupil_radius, color=(0, 0, 255), thickness=1)
    # cv2.circle(img=image_bgr, center=(cX, cY), radius=new_pupil_radius, color=(0, 0, 255), thickness=1)
    # cv2.circle(img=image_bgr, center=(cX, cY), radius=iris_radius, color=(0, 0, 255), thickness=1)
    # cv2.circle(img=image_bgr, center=(cX, cY), radius=new_iris_radius, color=(0, 0, 255), thickness=1)
    # cv2.imshow("image_bgr", image_bgr)

    # # ROI 그리기
    # pupil_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_pupil_radius, color=(255, 255, 255), thickness=-1)
    # iris_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_iris_radius, color=(255, 255, 255), thickness=-1)
    # total_segmentation = np.where(pupil_segmentation == 255, 0, iris_seg)
    # total_segmentation = np.where(iris_segmentation == 255, total_segmentation, 0)
    # cv2.imshow("total_segmentation", total_segmentation)

    # ############ 30도씩 섹터별로 나누기
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
    #     result = cv2.hconcat([sector, result_image])
    #     cv2.imshow("result_"+str(i), result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    ############ 30도씩 섹터별로 엔트로피 구하기
    normalized_entropys = []
    normalized_entropys_noLed = []

    for i in range(12):
        sector = image_bgr.copy()
        cv2.imshow("s!!!!!!", sector)

        startAngle = 270 + i * 30
        endAngle = startAngle + 30
        cv2.ellipse(img=sector, center=(cX, cY), axes=(iris_radius, iris_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(255, 0, 0), thickness=-1)
        cv2.ellipse(img=sector, center=(cX, cY), axes=(pupil_radius, pupil_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(0, 0, 0), thickness=-1)
        image_sector = np.where(sector == (255, 0, 0), image_bgr, 0)
        cv2.imshow("sector!!!!!!!!!!" , image_sector)


        # 1> 특정각도에서 홍채이미지
        image_sector_gray = cv2.cvtColor(image_sector, cv2.COLOR_BGR2GRAY)
        cv2.imshow("ssss!!!!!!", sector)
        img_sector = np.where(sector == (255, 0, 0), image_rgb, 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 2> 특정각도에서 홍채이미지의 엔트로피 값
        img_entropy_sector = entropy(image_sector_gray, disk(8))
        # 3> 특정각도에서 홍채이미지의 엔트로피의 임계치 이상의 값
        threshold_entropy_sector = np.where(img_entropy_sector > 5.5, 255, 0)
        normalized_entropy = np.sum(threshold_entropy_sector/255)/segmentation_area
        normalized_entropys.append(round(normalized_entropy * 12))

        #### led 비취는 부분 제거 한 gray image
        # 1> 특정각도에서 홍채이미지
        image_sector_gray_noLed = np.where(image_sector_gray > 170, 0, image_sector_gray)
        # 2> 특정각도에서 홍채이미지의 엔트로피 값
        img_entropy_sector_noLed = entropy(image_sector_gray_noLed, disk(8))
        # 3>특정각도에서 홍채이미지의 엔트로피의 임계치 이상의 값
        threshold_entropy_sector_noLed = np.where(img_entropy_sector_noLed > 5.5, 255, 0)
        normalized_entropy_noLed = np.sum(threshold_entropy_sector_noLed/255)/segmentation_area
        normalized_entropys_noLed.append(round(normalized_entropy_noLed * 12))

        # 위의 3개의 이미지를 그래프로 그리기
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        ax0.imshow(img_sector)
        ax0.set_title("image")

        threshold_entropy_sector = ax1.imshow(threshold_entropy_sector)
        ax1.set_title("Local 8(>5.5):  " + str(round(normalized_entropy*12)))

        img_entropy_sector = ax2.imshow(img_entropy_sector, cmap='viridis')
        ax2.set_title("Local entropy 8")
        ax2.axis("off")
        fig.colorbar(img_entropy_sector, ax=ax2)
        fig.tight_layout()
        plt.show()

    ############ entropy 분석 ############
    fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(nrows=2, ncols=5, figsize=(16, 6))
    ax0.imshow(image_rgb)
    ax0.set_title("roi image")

    ax1.imshow(image_gray)
    ax1.set_title("gray")

    entr_img_2 = entropy(image_gray, disk(8))
    entr_img_2 = ax2.imshow(entr_img_2, cmap='viridis')
    ax2.set_title("Local entropy 8")
    ax2.axis("off")
    fig.colorbar(entr_img_2, ax=ax2)

    entr_img_3 = entropy(image_gray, disk(8))
    entr_img_3 = np.where(entr_img_3 > 5.5, 255, 0)
    normalized_entropy = np.sum(entr_img_3/255)/segmentation_area
    entr_img_3 = ax3.imshow(entr_img_3)
    ax3.set_title("Local 8(>5.5):  " + str(round(normalized_entropy)))

    ########
    ax5.imshow(image_rgb_noLed)
    ax5.set_title("roi image")

    ax6.imshow(image_gray_noLed)
    ax6.set_title("image_threshold")

    entr_img_2 = entropy(image_gray_noLed, disk(8))
    entr_img_2 = ax7.imshow(entr_img_2, cmap='viridis')
    ax7.set_title("Local entropy 8")
    ax7.axis("off")
    fig.colorbar(entr_img_2, ax=ax7)

    entr_img_3 = entropy(image_gray_noLed, disk(8))
    entr_img_3 = np.where(entr_img_3 > 5.5, 255, 0)
    normalized_entropy = np.sum(entr_img_3 / 255) / segmentation_area
    entr_img_3 = ax8.imshow(entr_img_3)
    ax8.set_title("Local 8(>5.5):  " + str(round(normalized_entropy)))

    ########
    x = np.arange(1, 13)
    ax4.plot(x, normalized_entropys)
    ax4.set_xlim(1, 12)
    ax4.set_ylim(0, 2500)

    x = np.arange(1, 13)
    ax9.plot(x, normalized_entropys_noLed)
    ax9.set_xlim(1, 12)
    ax9.set_ylim(0, 2500)

    fig.tight_layout()
    plt.show()
    new_image_path = image_path[:-4]+"_plt.png"
    # plt.savefig(new_image_path)

