import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

label_dir = '/home/chungsong/3DCV/SMOKE/tools/logs_AddDepth2/inference/kitti_demo/data/'
image_dir = '/home/chungsong/3DCV/datasets/demo/image_2/'
calib_dir = '/home/chungsong/3DCV/datasets/demo/calib/'
result_dir = '/home/chungsong/3DCV/datasets/demo/results/'

dataset = [name.split('.')[0] for name in sorted(os.listdir(image_dir))]

all_image = sorted(os.listdir(image_dir))

for f in all_image:
    print(f)
    image_file = image_dir + f
    calib_file = calib_dir + f.replace('png', 'txt')
    label_file = label_dir + f.replace('png', 'txt')

    # read calibration data
    for line in open(calib_file):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3,4))
        
    image = cv2.imread(image_file)
    cars = []

    # Draw 3D Bounding Box
    for line in open(label_file):
        line = line.strip().split(' ')
        
        object_type = line[0]
        occluded = line[2]
        dims   = np.asarray([float(number) for number in line[8:11]])
        center = np.asarray([float(number) for number in line[11:14]])
        rot_y  = float(line[3]) + np.arctan(center[0]/center[2])#float(line[14])

        # Only draw car and fully or partially visiable
        if(object_type != 'Car' or occluded not in ['0', '1']):
            continue

        box_3d = []

        for i in [1,-1]:
            for j in [1,-1]:
                for k in [0,1]:
                    point = np.copy(center)
                    point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
                    point[1] = center[1] - k * dims[0]

                    point = np.append(point, 1)
                    point = np.dot(cam_to_img, point)
                    point = point[:2]/point[2]
                    point = point.astype(np.int16)
                    box_3d.append(point)

        # line_color = np.random.randint(0, 255, size=3)
        # line_color = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
        line_color = (0, 0, 255)

        for i in range(4):
            point_1_ = box_3d[2*i]
            point_2_ = box_3d[2*i+1]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), line_color, 2)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i+2)%8]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), line_color, 2)
                
    cv2.imwrite(result_dir+f, image)