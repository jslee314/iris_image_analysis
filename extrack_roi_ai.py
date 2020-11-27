from seg_master import UNet
import glob
import cv2
import numpy as np
from PIL import Image
from find_radius.find_pupil_radius import get_radius, get_centor, undesired_objects

src_path = "D:/2. data/total_iris/original_400/"
dst_path = "D:/2. data/total_iris/only_iris_40/"
image_paths = sorted(glob.glob(src_path + "*.png"))

images_rgb_np = []
images_bgr = []
image_names = []

size = 640, 480
for i, image_path in enumerate(image_paths):
    image_names.append(image_path.split("\\")[-1])
    image_rgb = Image.open(image_path).convert("RGB")
    image_rgb = image_rgb.resize(size, resample=Image.BICUBIC)

    # convert bgr
    r, g, b = image_rgb.split()
    image_bgr = Image.merge("RGB", (b, g, r))
    images_bgr.append(image_bgr)

    image_rgb = np.array(image_rgb)
    images_rgb_np.append(image_rgb)
rgb_dataset = np.array(images_rgb_np, dtype="float") / 255.0

m = UNet.UNet(nClasses=1, input_height=480, input_width=640)

pupil_h5model_name = "seg_master/unet_pupil_weight.h5"
m.load_weights(pupil_h5model_name)
pupil_predicts = m.predict(rgb_dataset, batch_size=2)

iris_h5model_name = "seg_master/unet_iris_weight.h5"
m.load_weights(iris_h5model_name)
iris_predicts = m.predict(rgb_dataset, batch_size=2)

# for image_rgb, pupil_predict, iris_predict, image_bgr, image_name in zip(images_rgb_np, pupil_predicts, iris_predicts, images_bgr, image_names):
for pupil_predict, iris_predict, image_bgr, image_name in zip(pupil_predicts, iris_predicts, images_bgr, image_names):
    pupil_seg = np.where(pupil_predict > 0.5, 0, image_bgr)
    iris_seg = np.where(iris_predict > 0.5, pupil_seg, 0)

    ####### pupil segmentation #######
    pupil_segmentation = np.where(pupil_predict > 0.5, 255, 0)
    pupil_lcc = undesired_objects(pupil_segmentation)
    pupil_lcc = pupil_lcc.astype(np.uint8)
    # cv2.imshow("pupil", pupil_lcc)

    ####### iris segmentation #######
    iris_segmentation = np.where(iris_predict > 0.5, 255, 0)
    iris_lcc = undesired_objects(iris_segmentation)
    iris_lcc = iris_lcc.astype(np.uint8)
    # cv2.imshow("iris", iris_lcc)

    # iris의 중심 구하기
    cX, cY = get_centor(pupil_lcc)

    # pupil/iris의 반지름 구하기
    pupil_radius = get_radius(pupil_lcc, cX, cY)
    iris_radius = get_radius(iris_lcc, cX, cX)

    # 반지름 조정하기
    btw_radius_ratio = round((iris_radius - pupil_radius) / 20)
    new_pupil_radius = pupil_radius + btw_radius_ratio
    new_iris_radius = iris_radius - (8 * btw_radius_ratio) # 4

    # 예측원과 중심 그리기
    image_bgr = np.array(image_bgr)

    cv2.circle(img=image_bgr, center=(cX, cY), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(img=image_bgr, center=(cX, cY), radius=pupil_radius, color=(0, 0, 255), thickness=1)
    cv2.circle(img=image_bgr, center=(cX, cY), radius=new_pupil_radius, color=(0, 0, 255), thickness=1)

    cv2.circle(img=image_bgr, center=(cX, cY), radius=iris_radius, color=(0, 0, 255), thickness=1)
    cv2.circle(img=image_bgr, center=(cX, cY), radius=new_iris_radius, color=(0, 0, 255), thickness=1)

    # cv2.imshow("image_bgr", image_bgr)

    # ROI 그리기
    pupil_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_pupil_radius, color=(255, 255, 255), thickness=-1)
    iris_segmentation = cv2.circle(img=np.zeros((480, 640, 3)), center=(cX, cY), radius=new_iris_radius, color=(255, 255, 255), thickness=-1)

    total_segmentation = np.where(pupil_segmentation == 255, 0, iris_seg)
    total_segmentation = np.where(iris_segmentation == 255, total_segmentation, 0)
    segmentation_area = np.sum(np.where(total_segmentation != 0, 1, 0))
    # cv2.imshow("total_segmentation", total_segmentation)

    # image crop
    y = (cY-new_iris_radius) if (cY-new_iris_radius) > 0 else 0
    x = (cX-new_iris_radius) if (cX-new_iris_radius) > 0 else 0
    total_segmentation = total_segmentation[y: cY+new_iris_radius, x:cX+new_iris_radius]
    cX = cX - x
    cY = cY - y

    image_name = image_name[:-4] + "_"+ str(segmentation_area) + "_" + str(cX) + "_" + str(cY) + "_" + str(new_pupil_radius) + "_" + str(new_iris_radius) + ".png"
    cv2.imwrite(dst_path + image_name, total_segmentation)

    # sectors_region = []
    # sectors_image = []
    # for i in range(12):
    #     sector = total_segmentation.copy()
    #     startAngle = 270 + i * 30
    #     endAngle = startAngle + 30
    #     cv2.ellipse(img=sector, center=(cX, cY), axes=(new_iris_radius, new_iris_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(255, 0, 0), thickness=-1)
    #     cv2.ellipse(img=sector, center=(cX, cY), axes=(new_pupil_radius, new_pupil_radius), angle=0, startAngle=startAngle, endAngle=endAngle, color=(0, 0, 0), thickness=-1)
    #     sectors_region.append(sector)
    #
    #     result_image = np.where(sector == (255, 0, 0), total_segmentation, 0)
    #     sectors_image.append(sector)
    #     # result = cv2.hconcat([sector, result_image])
    #     # cv2.imshow("result_"+str(i), result)

    # # 방법 1: SIFT(Scale Invariant Feature Transform): 크기불변 이미지 특성 검출
    # gray_sector = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
    # sector2, sector3 = None, None
    #
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray_sector, None)
    #
    # sector2 = cv2.drawKeypoints(gray_sector, kp, sector2)
    # sector3 = cv2.drawKeypoints(gray_sector, kp, sector3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # result = cv2.hconcat([sector2, sector3])

    # # 방법 2:
    # gray_sector = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray_sector], [0], None, [256], [0, 256])
    #
    #
    #
    # cv2.imshow("result_"+str(i), hist)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


