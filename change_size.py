# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:17:23 2018

@author: sds
"""
#change your image's size to 299*299
#use opencv2
import cv2
import os

def alter(path,object):
    s = os.listdir(path)
    count = 1
    for i in s:
        document = os.path.join(path,i)
        img = cv2.imread(document)
        img = cv2.resize(img, (299,299))
        listStr = [str(count)]
        fileName = ''.join(listStr)
        cv2.imwrite(object+os.sep+'%s.jpg' % fileName, img)
        count = count + 1

alter('E:/yun/baozhuangcengjiplus','E:/yun/baozhuangcengji')

#use PIL
'''
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=299,height=299):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("E:/yun/juanjiplus/*.jpg"):
    convertjpg(jpgfile,"E:/yun/juanji")
''' 