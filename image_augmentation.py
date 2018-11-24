# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:10:10 2018

@author: sds
"""
import cv2
import numpy as np
'''
定义裁剪函数，四个参数是左上角横坐标x0,左上角纵坐标y0,裁剪宽度w,裁剪高度h。
'''
crop_image=lambda img ,x0,y0,w,h:img[y0:y0+h,x0:x0+w]

def random_crop(img,area_ratio,hw_vari):
    h,w=img.shape[:2]
    hw_delta=np.random.uniform(-hw_vari,hw_vari)
    hw_mult=1+hw_delta
    w_crop=int(round(w*np.sqrt(area_ratio*hw_mult)))
    if w_crop>w:
        w_crop=w
    h_crop=int(round(h*np.sqrt(area_ratio/hw_mult)))
    if h_crop>h:
        h_crop=h
    x0=np.random.randint(0,w-w_crop+1)
    y0=np.random.randint(0,h-h_crop+1)
    return crop_image(img,x0,y0,w_crop,h_crop)
'''
定义旋转函数：angle是旋转角度，crop是布尔值表明是否裁剪去除黑边
'''
def rotate_image(img,angle,crop):
    h,w=img.shape[:2]
    angle%=360
    M_rotate=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img_rotated=cv2.warpAffine(img,M_rotate,(w,h))
    if crop:
        angle_crop=angle%180
        if angle_crop>90:
            angle_crop=180-angle_crop
        theta=angle_crop*np.pi /180.0
        hw_ratio=float(h)/float(w)
        tan_theta=np.tan(theta)
        numerator=np.cos(theta)+np.sin(theta)*tan_theta
        r=hw_ratio if h>w else 1/hw_ratio
        denominator=r*tan_theta+1
        crop_mult=numerator/denominator
        w_crop=int(round(crop_mult*w))
        h_crop=int(round(crop_mult*h))
        x0=int((w-w_crop)/2)
        y0=int((h-h_crop)/2)
        img_rotated=crop_image(img_rotated,x0,y0,w_crop,h_crop)
    return img_rotated


'''
随机旋转angle_vari是旋转角度范围，p_crop是要进行去黑边裁剪的比例
'''
def random_ratate(img,angle_vari,p_crop):
    angle=np.random.uniform(-angle_vari,angle_vari)
    crop=False if np.random.random()>p_crop else True
    return rotate_image(img,angle,crop)


'''
定义hsv变换函数：hue_delta是色调变化比例，sat_delta是饱和度变化比例，val_delta是明度变化比例
'''
def hsv_transform(img,hue_delta,sat_mult,val_mult):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_delta)%180
    img_hsv[:,:,1]*=sat_mult
    img_hsv[:,:,2]*=val_mult
    img_hsv[img_hsv>255]=255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8),cv2.COLOR_HSV2BGR)


'''
进行hsv变换,hue_vari是色调变化比例范围
sat_vari是饱和度变化比例范围，val_vari是明度变化比例范围
'''
def random_hsv_transform(img,hue_vari,sat_vari,val_vari):
    hue_delta=np.random.randint(-hue_vari,hue_vari)
    sat_mult=np.random.uniform(-sat_vari,sat_vari)+1
    val_mult=np.random.uniform(-val_vari,val_vari)+1
    return hsv_transform(img,hue_delta,sat_mult,val_mult)


'''
定义gamma变换函数
'''
def gamma_transform(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

'''
随机gamma变换，gamma_vari是Gamma的变化范围
'''
def random_gamma_transform(img,gamma_vari):
    log_gamma_vari=np.log(gamma_vari)
    alpha=np.random.uniform(-log_gamma_vari,log_gamma_vari)
    gamma=np.exp(alpha)
    return gamma_transform(img,gamma)
    





            
            
            