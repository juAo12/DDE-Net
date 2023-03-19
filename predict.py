
import time

import os
import cv2
import numpy as np
from PIL import Image

from centernet import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    #----------------------------------------------------------------------------------------------------------#
    #   mode:'predict', 'fps'
    #----------------------------------------------------------------------------------------------------------#
    # mode = "predict"
    mode = "fps"
    crop            = False
    count           = False
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    
    if mode == "predict":

        # while True:
        #     img = input('Input image filename:')
        #     try:
        #         image = Image.open(img)
        #     # try:
        #         # image = Image.open('/newdisk/homee/yangao1/dataset/VOCdevkit/VOC2007/JPEGImages/000051.jpg')
        #     except:
        #         print('Open Error! Try again!')
        #         continue
        #     else:
        #         r_image = centernet.detect_image(image, crop=crop, count=count)
        #         r_image.save("img.jpg", dir_save_path=dir_save_path)
        #         # r_image.show()

        dir_origin_path = "img/"
        files = os.listdir(dir_origin_path)

        for img in files:
            image = Image.open("img/" + img)
            r_image = centernet.detect_image(image, crop=crop, count=count)
            r_image.save(str(img))
            print("img/" + img)
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
