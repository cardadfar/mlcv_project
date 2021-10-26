
#TODO: none of this works. would be nice to have...

import cv2
import numpy as np
import glob

frameSize = (28, 28)

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 1, frameSize)

for filename in glob.glob('plots/test/*.png'):
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()

cap = cv2. VideoCapture("output.avi")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

