from itertools import count
import skimage.io as sio
import os
import numpy as np
import pickle as pickle
import torch
import numpy as np
import time
import os
import math
import scipy.io
import cv2

classes_colour =[
                [0,     0,   0],
                [128,   0,   0],
                [  0, 128,   0],
                [128, 128,   0],
                [  0,   0, 128],
                [128,   0, 128],
                [  0, 128, 128],
                [128, 128, 128],
                [ 64,   0,   0],
                [192,   0,  0],
                [ 64, 128,   0],
                [192, 128,   0],
                [ 64,   0, 128],
                [192,   0, 128],
                [ 64, 128, 128],
                [192, 128, 128],
                [  0,  64,   0],
                [128,  64,   0],
                [  0, 192,   0],
                [128, 192,   0],
                [  0,  64, 128]]

def get_file_ids(id_path):
    file_ids = open(id_path)
    ids = []
    for line in file_ids.readlines():
        ids.append(line[:-1])
    return ids

def img_to_2d_test(img):
    img_2d = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i, colour in enumerate(classes_colour):
        m = np.all(img==np.array(colour), axis=2)
        img_2d[m] = i
    return img_2d

def tensor_maker(cur_batch, predict_loc,gt_loc,gpu=True):
    #returns the tensors of GT and IMG(use 513 and crops)
    #########
    post_fix = '.png'
    #########
    batch_size = len(cur_batch)
    height = 513
    width = 513
    prediction_tensor = np.full((batch_size,height,width), 255)
    gt_tensor = np.full((batch_size,height,width), 255)
    counter = 0
    for id in cur_batch:
        img = cv2.imread(os.path.join(predict_loc,id+post_fix))
        gray = img_to_2d_test(img)
        prediction_tensor[counter,:img.shape[0],:img.shape[1]] = np.copy(gray)
        img = cv2.imread(os.path.join(gt_loc,id+post_fix))
        gray = img_to_2d_test(img)
        gt_tensor[counter,:img.shape[0],:img.shape[1]] = np.copy(gray)
        counter +=1

    prediction_tensor = torch.from_numpy(prediction_tensor).long()
    gt_tensor = torch.from_numpy(gt_tensor).long()

    #if(gpu):
    #    prediction_tensor = prediction_tensor.cuda()
    #    gt_tensor = gt_tensor.cuda()
    return (prediction_tensor,gt_tensor)

def hist_per_batch(tensor_1, tensor_2, ignore_label=255, classes=21):
    hist_tensor = torch.zeros(classes,classes)
    for class_2_int in range(classes):
        tensor_2_class = torch.eq(tensor_2, class_2_int).long()
        for class_1_int in range(classes):
            tensor_1_class = torch.eq(tensor_1, class_1_int).long()
            tensor_1_class = torch.mul(tensor_2_class,tensor_1_class)
            count = torch.sum(tensor_1_class)
            hist_tensor[class_2_int,class_1_int] +=count

    return hist_tensor

def hist_maker(predict_loc,gt_loc,file_id_list,batch_size= 20,ignore_label = 255, classes=21,gpu=True):
    hist_tensor = torch.zeros(classes,classes)
    max_iter = int(math.ceil(len(file_id_list)/batch_size)+1)
    for i in range(max_iter):
        cur_batch = file_id_list[batch_size*i:min(len(file_id_list),batch_size*(i+1))]
        predict_tensor, gt_tensor = tensor_maker(cur_batch, predict_loc,gt_loc,gpu = gpu)
        hist_batch = hist_per_batch(predict_tensor, gt_tensor, ignore_label=255, classes=21)
        hist_tensor = torch.add(hist_tensor,hist_batch)
    return hist_tensor

def mean_iou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])+np.sum(hist_matrix[:,i])-hist_matrix[i,i]))
        print('class',class_names[i],'miou',class_scores[i])
    print('Mean IOU:',np.mean(class_scores))
    return class_scores

def mean_pixel_accuracy(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
        print('class',class_names[i],'mean_pixel_accuracy',class_scores[i])
    return class_scores

def pixel_accuracy(hist_matrix):
    num = np.trace(hist_matrix)
    p_a =  num/max(1,np.sum(hist_matrix).astype('float'))
    print('Pixel accuracy:',p_a)
    return p_a

def freq_weighted_miou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = np.sum(hist_matrix[i,:])*hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
    fmiou = np.sum(class_scores)/np.sum(hist_matrix).astype('float')
    print('Frequency Weighted mean accuracy:',fmiou)
    return fmiou
