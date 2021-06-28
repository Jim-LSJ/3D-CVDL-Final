import os, glob
import cv2
import numpy as np

img_paths = '/home/chungsong/3DCV/datasets/demo/results'

img_names = glob.glob(os.path.join(img_paths, '*.png'))
img_names.sort(key=lambda x:int(x[-7:-4]))

fps = 10
size = (1242, 375)
output_path = './video.mp4'
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter(output_path, fourcc, fps, size)

## make video
for img_name in img_names:
    rimg = cv2.imread(img_name)
    out.write(rimg)
    
out.release()