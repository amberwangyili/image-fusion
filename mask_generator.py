#!/usr/bin/env python

import numpy as np
import cv2
from os import path



class Painter():
    def __init__(self,source_path, target_path):
        self.source = cv2.imread(source_path,cv2.IMREAD_COLOR)
        self.target = cv2.imread(target_path,cv2.IMREAD_COLOR)

        self.img = self.source.copy()
        self.mask = np.zeros(self.source.shape)
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.name = 'Mask generator s:save; q:quit'


    def draw_mask(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.rectangle(self.mask, (self.ix-4, self.iy-4), (x+4, y+4),(255, 255, 255), -1)
                cv2.rectangle(self.img,(self.ix-4,self.iy-4),(x+4,y+4),(0,255,255),-1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    def gen_mask(self):

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name,self.draw_mask)

        while(1):
            cv2.imshow(self.name,self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                cv2.destroyAllWindows()
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                break
        return self.mask



class Mover():
    def __init__(self,target_path,mask):
        self.image_path = target_path
        self.move = False
        self.first = True
        self.x0 = 0
        self.y0 = 0
        self.xi = 0
        self.yi = 0
        
        self.image = cv2.imread(target_path)
        self.image_ = self.image.copy()
        
        self.origin_mask = mask
        self.origin_mask_ = np.zeros(self.image.shape)
        self.origin_mask_[np.where(self.origin_mask!= 0)] = 255
        
        self.mask = self.origin_mask_.copy()
        self.name = 'Move mask: s--save, q--quit' 
        
    def blend_patch(self,image,mask):
        patch = image.copy()
        alpha = 0.3
        patch[mask!=0] = patch[mask!=0]*alpha + 255*(1-alpha)
        blend_patch = patch.astype(np.uint8)
        return blend_patch
    

    def move_mask(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.move = True
            if self.first:
                self.x0,self.y0 = x,y
                self.first = False
            self.xi, self.yi = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.move:
                M = np.float32([[1,0,x-self.xi],[0,1,y-self.yi]])

                self.mask = cv2.warpAffine(self.mask,M,(self.mask.shape[1],self.mask.shape[0]))
                cv2.imshow(self.name,self.blend_patch(self.image,self.mask))
                self.xi,self.yi = x,y
        elif event == cv2.EVENT_LBUTTONUP:
            self.move = False
    
    def gen_mask(self):
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name,self.move_mask)
        
        while True:
            cv2.imshow(self.name,self.blend_patch(self.image,self.mask))
            key = cv2.waitKey(1)&0xFF

            if key == ord('s'):
                break
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit()
        roi = self.mask
        cv2.imshow('press s to save the mask',roi)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        return self.xi - self.x0, self.yi - self.y0, self.mask

    
    
    
 
 
# def main():

#     test_dir = './test'
#     S = 'source.png'
#     T = 'target.png'

#     source = cv2.imread(path.join(test_dir,S))
#     target = cv2.imread(path.join(test_dir,T))
#     painter = Painter(path.join(test_dir,S),path.join(test_dir,T))
#     target_mask = painter.gen_mask()

#     mover = Mover(path.join(test_dir,T),target_mask)

#     offset_x, offset_y, target_mask = mover.gen_mask()

#     offset = (offset_x,offset_y)

#     M =np.float32([[1,0,offset[0]],[0,1,offset[1]]])
#     y_max, x_max = target.shape[:-1]
#     y_min, x_min = 0,0
#     x_range = x_max - x_min
#     y_range = y_max - y_min
#     source = cv2.warpAffine(source,M,(x_range,y_range))
#     cv2.imwrite(path.join(test_dir,'transformed_mask.png'),target_mask)
#     cv2.imwrite(path.join(test_dir,'transformed_source.png'),source)
# if __name__ == '__main__':
# 	main()
	