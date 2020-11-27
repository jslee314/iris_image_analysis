from seg_master import UNet
import glob
import cv2
import numpy as np
from PIL import Image

src_path = "D:/2. data/total_iris/original_4/"
dst_path = "D:/2. data/total_iris/only_iris_4/"
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

for image_rgb, pupil_predict, iris_predict, image_name in zip(images_rgb_np, pupil_predicts, iris_predicts, image_names):

    pupil_segmentation = np.where(pupil_predict > 0.5, 100, image_rgb)
    iris_segmentation_6 = np.where(iris_predict > 0.6, pupil_segmentation, 100)
    iris_segmentation_5 = np.where(iris_predict > 0.6, pupil_segmentation, 100)


    # Image._show(Image.fromarray(image_rgb))
    # Image._show(Image.fromarray(iris_segmentation))
    # filename = image_name[:-4] +  +".png"
    Image.fromarray(iris_segmentation_5).save(dst_path + "" + image_name)

