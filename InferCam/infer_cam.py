
from ast import arg
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from scipy import ndimage
import pickle
from datetime import datetime
import time

from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import logging
import torch
import pptk as pptker
import itertools
# from network import resnet38_cls_sccam


classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa', 'train','tvmonitor']

classes_colour =[
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

def get_pca_line(X):
    pca = PCA(n_components=3)
    data_cluster = np.array(X)
    
    #Remove masked values
    data_cluster=data_cluster[np.any(data_cluster!=0,axis=-1)] 
    
    pca.fit(data_cluster)
    
    line = pca.components_[0]
    centroid = pca.mean_
    centroid = np.mean(data_cluster, axis = 0)
    variance = pca.explained_variance_[-1] + pca.explained_variance_[-2]
    
    return line, centroid

def get_distance_line_gpu(X_gpu, planes, centroids, outliers=0):
        X_gpu = torch.from_numpy(X_gpu).float().to("cuda:0")
        lines4d_gpu = torch.from_numpy(planes).float().to("cuda:0")
        centroids_gpu = torch.from_numpy(centroids).float().to("cuda:0")
        distance = torch.zeros((len(centroids), X_gpu.shape[0]))
        
        verctors4d_gpu = - X_gpu + centroids_gpu
        
        temp = torch.cross(verctors4d_gpu, torch.cat(X_gpu.shape[0]*[lines4d_gpu.unsqueeze(0)]), dim=1)
        numerator = torch.norm(temp, p=2, dim=1)
        denominator = torch.norm(lines4d_gpu)
        distance = numerator/denominator
        return distance

def infer_split_cam(args):
    model = getattr(importlib.import_module(args.network), 'Net')()

    print(torch.cuda.is_available())
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count()
    model_replicas = [model]
    #model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    infer_dataset = voc12.data.VOC12ClsDatasetMSFsplit(args.infer_list, voc12_root=args.voc12_root, 
                                                        aug_path = args.split_path, scales=(1, 0.5, 1.5, 2.0),
                                                        inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # cam_mask_dict = {}
    for iter, (img_name, output_list, label) in enumerate(infer_data_loader):
        #if (iter <= 1379):
        #    continue
        img_name = img_name[0]; label = label[0]
        cam_dict = {}
        for split_index, img_list in enumerate(output_list):
            if split_index == 0:   # original image
                orig_img = cv2.imread(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name))
                raw_img = orig_img
                # raw_img = np.asarray(Image.open(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name)))
                cam_mask = np.zeros((orig_img.shape[0], orig_img.shape[1], 4, 20))
                orig_img_size = orig_img.shape[:2]
                cam_matrix = np.zeros((20, orig_img_size[0], orig_img_size[1]))
                last_left_area_matrix = np.ones((orig_img_size[0], orig_img_size[1]))
                pixel_sum = orig_img_size[0] * orig_img_size[1]
            else: # each split
                aug_img_dir = args.split_path
                orig_img = cv2.imread(os.path.join(aug_img_dir, '{}_{}.jpg'.format(img_name, split_index)))
                orig_img_size = orig_img.shape[:2]
                cam_matrix = np.zeros((20, orig_img_size[0], orig_img_size[1]))
                last_left_area_matrix = np.ones((orig_img_size[0], orig_img_size[1]))
                pixel_sum = orig_img_size[0] * orig_img_size[1]
                orig_img_size = orig_img.shape[:2]

                split_start = datetime.now()

                for dropout_index in range(5):
                    start = datetime.now()
                    def _work(i, img):
                        with torch.no_grad():
                            with torch.cuda.device(i%n_gpus):
                                cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                                cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                                if i % 2 == 1:
                                    cam = np.flip(cam, axis=-1)
                                return cam

                    if dropout_index > 0:
                        new_img_list = []
                        for img_index, img in enumerate(img_list):
                            _, _, h, w = img.shape
                            left_area_matrix = F.interpolate(left_area_matrix, (h, w))
                            rgb_mean = torch.mean(img, dim=(2,3))
                            mask_mean = torch.ones_like(img)
                            mask_mean[:, 0, :, :] = mask_mean[:,0,:,:] * rgb_mean[0,0]
                            mask_mean[:, 1, :, :] = mask_mean[:,1,:,:] * rgb_mean[0,1]
                            mask_mean[:, 2, :, :] = mask_mean[:,2,:,:] * rgb_mean[0,2]
                            # 
                            # mask_mean[:, 0, :, :] = mask_mean[:, 0, :, :] * (122.675 / 255)
                            # mask_mean[:, 1, :, :] = mask_mean[:, 1, :, :] * (116.669 / 255)
                            # mask_mean[:, 2, :, :] = mask_mean[:, 2, :, :] * (104.008 / 255)

                            new_img = ((left_area_matrix * img) + (1-left_area_matrix) * mask_mean).float()
                            new_img_list.append(new_img)
                    else:
                        new_img_list = img_list

                    thread_pool = pyutils.BatchThreader(_work, list(enumerate(new_img_list)),
                                                        batch_size=12, prefetch_size=0, processes=args.num_workers)

                    cam_list = thread_pool.pop_results()

                    sum_cam = np.sum(cam_list, axis=0)
                    norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

                    cam_matrix = np.stack((cam_matrix, norm_cam), axis=3)
                    cam_matrix = np.max(cam_matrix, axis=3)

                    # left_area_matrix = 1 - np.max(cam_matrix, axis=0)
                    left_area_matrix = ((np.max(cam_matrix, axis=0) < 0.7) * 1)
                    cam_mask_diff = (np.sum(last_left_area_matrix) - np.sum(left_area_matrix)) / pixel_sum
                    last_left_area_matrix = left_area_matrix
                    left_area_matrix = torch.from_numpy(left_area_matrix).unsqueeze(0).unsqueeze(0).float()
                    if cam_mask_diff < 0.02:
                        break

            if split_index > 0:
                for i in range(20):
                    if label[i] > 1e-5:
                        if split_index == 1:
                            cam_mask[0:orig_img_size[0], 0:orig_img_size[1], 0, i] = cam_matrix[i]
                        elif split_index == 2:
                            cam_mask[0:orig_img_size[0], cam_mask.shape[1]-orig_img_size[1]:cam_mask.shape[1], 1, i] = cam_matrix[i]
                        elif split_index == 3:
                            cam_mask[cam_mask.shape[0]-orig_img_size[0]:cam_mask.shape[0], 0:orig_img_size[1], 2, i] = cam_matrix[i]
                        elif split_index == 4:
                            cam_mask[cam_mask.shape[0]-orig_img_size[0]:cam_mask.shape[0], cam_mask.shape[1]-orig_img_size[1]:cam_mask.shape[1], 3, i] = cam_matrix[i]
                            cam_dict[i] = np.max(cam_mask[:,:,:,i], 2)

            if split_index == 4:
                if args.heatmap is not None:
                    img = raw_img
                    keys = list(cam_dict.keys())
                    for target_class in keys:
                        mask = cam_dict[target_class]
                        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                        img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0] ))
                        cam_output = heatmap * 0.3 + img * 0.5

                        cv2.imwrite(os.path.join(args.heatmap, img_name + '_{}.jpg'.format(classes[target_class])), cam_output)
                
                if args.colour_seg is not None:
                    img = raw_img
                    X = img.reshape((-1, 3))
                    keys = list(cam_dict.keys())

                    anti_masks = {}
                    class_masks = {}

                    for temp_class in keys:
                        temp_mask = cam_dict[temp_class]
                        anti_masks[temp_class] = temp_mask

                    blank_image = np.zeros( (raw_img.shape[0], raw_img.shape[1], 3), np.uint8)
                    cam_image = np.zeros( (raw_img.shape[0], raw_img.shape[1], 3), np.uint8)
                    for target_class in keys:
                        mask = cam_dict[target_class]
                        
                        anti_mask = None
                        for key, value in cam_dict.items():
                            if key != target_class:
                                if anti_mask is None:
                                    anti_mask = value
                                else:
                                    anti_mask = anti_mask + value

                        # 100, 255
                        
                        ### Get Mask
                        #th, dst = cv2.threshold(mask*255.0, 100, 255, cv2.THRESH_BINARY)
                        th, dst = cv2.threshold(mask, 0.7, 1.0, cv2.THRESH_BINARY)
                        dst = dst.astype(np.uint8)
                        res = cv2.bitwise_and(img, img, mask=dst)

                        #line, centroid = get_pca_line(res)
                        #cv2.imwrite(os.path.join(args.colour_seg + "Cam", img_name + '_{}.jpg'.format(classes[target_class])), res)

                        ### Apply grabCut
                        temp_img = img
                        newmask = dst
                        bgdModel = np.zeros((1,65),np.float64)
                        fgdModel = np.zeros((1,65),np.float64)
                        # wherever it is marked white (sure foreground), change mask=1
                        # wherever it is marked black (sure background), change mask=0
                        mask_cut = np.zeros(temp_img.shape[:2],np.uint8)
                        mask_cut[newmask == 0] = 2
                        # Add anti-mask
                        if anti_mask is not None:
                            th, anti_mask = cv2.threshold(anti_mask *255.0, 200, 255, cv2.THRESH_BINARY)
                            anti_mask  = anti_mask .astype(np.uint8)
                            res = cv2.bitwise_and(img, img, mask=anti_mask)
                            mask_cut[anti_mask != 0] = 0
                            mask_cut[anti_mask == 0] = 2
                            #cv2.imwrite(os.path.join(args.colour_seg + "Test", img_name + 'AntiMask_{}.jpg'.format(classes[target_class])), res)
                        mask_cut[newmask != 0] = 1
                        mask_cut, bgdModel, fgdModel = cv2.grabCut(temp_img,mask_cut,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                        mask_cut = np.where((mask_cut==2)|(mask_cut==0),0,1).astype('uint8')
                        test_img = temp_img*mask_cut[:,:,np.newaxis]

                        class_masks[target_class] = mask_cut
                        #cv2.imwrite(os.path.join(args.colour_seg, img_name + 'cut_{}.jpg'.format(classes[target_class])), test_img)
                        blank_image[class_masks[target_class] != 0] = classes_colour[target_class]
                        cam_image[dst != 0] = classes_colour[target_class]
                        # Show cluster
                        #print(line)
                        #print(centroid)
                        #print(img.shape)

                        #t = np.linspace(-255, 255, 1000)
                        #axes = np.array([t,t,t])

                        #line1 = np.array([new_planes*axes[:,i]+new_centroids for i in range(1000)])
                        #line2 = np.array([line*axes[:,i]+centroid for i in range(1000)])
                        #A = np.vstack((res.reshape((-1, 3)), *line2[:,])) #,line1[:,1]
                        #v = pptker.viewer(A)

                        #colours = get_distance_line_gpu(X, line, centroid)

                        #test = (colours.cpu()).numpy().reshape(img.shape[0:2])
                        #threshold = np.mean(test)
                        #th, dst = cv2.threshold(np.uint8(test), 0, 40, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                        #dst = dst.astype(np.uint8)
                        #res = cv2.bitwise_and(img, img, mask=dst)

                        #heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                        #img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0] ))
                        #cam_output = heatmap * 0.3 + img * 0.5

                        #cv2.imwrite(os.path.join(args.colour_seg, img_name + '_{}.jpg'.format(classes[target_class])), res)
                
                blank_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(args.colour_seg + "Test", img_name + 'Mask.png'), blank_image)
                cv2.imwrite(os.path.join(args.colour_seg + "Cam", img_name + 'Cam.png'), cam_image)

                raw_img = np.asarray(Image.open(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name)))

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        print(iter)

if __name__ == '__main__':
    total_start = datetime.now()

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--split_path", default='/home/users/u5876230/fbwss_output/baseline_trainaug_aug/', type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--heatmap", default=None, type=str)
    parser.add_argument("--colour_seg", default=None, type=str)

    args = parser.parse_args()
    infer_split_cam(args)

