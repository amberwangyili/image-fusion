#!/usr/bin/env python
from mask_generator import Painter, Mover
from poisson_edit import *
import cv2
import numpy as np
from os import path


def main():
    test_dir = './test'
    S = 'source.png'
    T = 'target.png'
    source = cv2.imread(path.join(test_dir,S))
    target = cv2.imread(path.join(test_dir,T))
    painter = Painter(path.join(test_dir,S),path.join(test_dir,T))
    target_mask = painter.gen_mask()
    mover = Mover(path.join(test_dir,T),target_mask)
    offset_x, offset_y, target_mask = mover.gen_mask()
    offset = (offset_x,offset_y)
    M =np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0,0
    x_range = x_max - x_min
    y_range = y_max - y_min
    source = cv2.warpAffine(source,M,(x_range,y_range))
    cv2.imwrite(path.join(test_dir,'transformed_mask.png'),target_mask)
    cv2.imwrite(path.join(test_dir,'transformed_source.png'),source)

    target = cv2.imread(path.join(test_dir,'target.png'),cv2.IMREAD_COLOR)
    source = cv2.imread(path.join(test_dir,'transformed_source.png'),cv2.IMREAD_COLOR)
    mask = cv2.imread(path.join(test_dir,'transformed_mask.png'),cv2.IMREAD_GRAYSCALE)


    composite = [poisson_edit_lib(source[:,:,i], target[:,:,i], mask) for i in range(3)]
    result = cv2.merge(composite)
    cv2.imwrite(path.join(test_dir, 'result.png'), result)


    #experiment
    # imglist =[]
    # for j in range(550,5050,50):
    #     composite = [poisson_edit(source[:,:,i], target[:,:,i], mask,j,'sor') for i in range(3)]
    #     result = cv2.merge(composite)
    #     imglist.append(result)
    #     cv2.imwrite(path.join(test_dir, 'gif/'+str(j)+'.png'), result)
    # import imageio

    # gif_path = './test/gif/sample_1.gif'
    # imageio.mimwrite(gif_path, np.asarray([i[:,:,::-1] for i in imglist]).astype(np.uint8), duration=0.04)
if __name__ == '__main__':
    main()

