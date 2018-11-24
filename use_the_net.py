# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:47:16 2018

@author: sds
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nets import inception
import os 
import time
#from datasets import imagenet
tf.reset_default_graph()
image_size=inception.inception_resnet_v2.default_image_size
#names=imagenet.create_readable_names_for_imagenet_labels()
slim=tf.contrib.slim
checkpoint_file='E:\yun\log2\model.ckpt-40000'

s=os.listdir('E:\\yun\\test\\goujuan')
sample_images=[]

for i in s:
    sample_images.append(i)
print(sample_images)

input_imgs=tf.placeholder("float",[None,image_size,image_size,3])
sess=tf.Session()
arg_scope=inception.inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
    logits,end_points=inception.inception_resnet_v2(input_imgs,is_training=False,num_classes=27)
saver=tf.train.Saver()
saver.restore(sess,checkpoint_file)
cloud={'Ac_cast':0,'Ac_cug':1,'Ac_flo':2,'Ac_lent':3,'Ac_op':4,'Ac_tra':5,'As_op':6,
       'As_tra':7,'Cb_calv':8,'Cb_cap':9,'Cc':10,'Ci_dens':11,'Ci_fil':12,'Ci_not':13,
       'Ci_unc':14,'Cs_fil':15,'Cs_nebu':16,'Cu_cong':17,'Cu_hum':18,'Fs':19,'Ns':20,
       'Sc_cast':21,'Sc_cug':22,'Sc_lent':23,'Sc_op':24,'Sc_tra':25,'St':26}

single_image=['9.jpg']
for image in single_image:
    reimg=Image.open(image).resize((image_size,image_size))
    reimg=np.array(reimg)
    reimg=reimg.reshape(-1,image_size,image_size,3)
    plt.figure()
    p1=plt.subplot(121)
    p2=plt.subplot(122)
    p1.imshow(reimg[0])
    p1.axis('off')
    p1.set_title("organization image")
    reimg_norm=2*(reimg/255.0)-1.0
    p2.imshow(reimg_norm[0])
    p2.axis('off')
    p2.set_title("input image")
    plt.show
    predict_values,logits_values=sess.run([end_points['Predictions'],logits],
                                          feed_dict={input_imgs:reimg_norm})
    index=np.argsort(predict_values)[:,-3:]
    out=predict_values[0,index[0,:]]
    #print(predict_values)
    for i in range(2,-1,-1):
        for key,val in cloud.items():
            if val==index[0,i]:
                print("可能性排名：",-i+3,key,out[i])
    print('_____________________________________________')
        
   
    #print(out[0])
    #print(index,np.max(logits_values))
    #print(index,np.argmax(logits_values))
    #print(np.argmax(predict_values),np.argmax(logits_values))
'''
for image in sample_images:
    image=os.path.join('E:\\yun\\test\\goujuan',image)
    reimg=Image.open(image).resize((image_size,image_size))
    reimg=np.array(reimg)
    reimg=reimg.reshape(-1,image_size,image_size,3)
    plt.figure()
    p1=plt.subplot(121)
    p2=plt.subplot(122)
    p1.imshow(reimg[0])
    p1.axis('off')
    p1.set_title("organization image")
    reimg_norm=2*(reimg/255.0)-1.0
    p2.imshow(reimg_norm[0])
    p2.axis('off')
    p2.set_title("input image")
    plt.show
    start=time.time()
    predict_values,logits_values=sess.run([end_points['Predictions'],logits],
                                          feed_dict={input_imgs:reimg_norm})
    index=np.argsort(predict_values)[:,-3:]
    out=predict_values[0,index[0,:]]
    #print(predict_values)
    _,image_name=os.path.split(image)
    print(image_name)
    for i in range(2,-1,-1):
        for key,val in cloud.items():
            if val==index[0,i]:
                print("可能性排名：",-i+3,key,out[i])
    end=time.time()
    print("用时：",end-start)
    print('_____________________________________________')    
'''    
    
    
    