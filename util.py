import cv2
import numpy as np
import os


def img2vid(img_dir, frame_size=(28, 28), output_path='video.avi'):
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, frame_size)
    filenames = os.listdir(img_dir)

    filenames = sorted(filenames, key=lambda x: int(x[:-4]))

    for filename in filenames:
        img = cv2.imread(img_dir + filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        video.write(img)

    video.release()


img2vid('results/test/')
