# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#%matplotlib inline

from generate_helper import *

def Generate_Sprites(img):
    points = get_key_points(img)
    masked = get_masked_image_rgb(img)

    edge = cv2.Canny(masked[:,:,3], 60, 180)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(edge, kernel, iterations=1)
    #nopixels =np.zeros(img_dilation.shape).astype(np.uint8)
    edges_rgb = cv2.merge((img_dilation,img_dilation, img_dilation))

    edges_copy = mark_points_edges(edges_rgb, points)
    masked_crop = crop_image(masked)
    #edges_crop = crop_image(edges_copy)

    coords = crop_coords(edges_copy)
    masked_crop  = masked[coords[0]:coords[1]+1, coords[2]:coords[3]+1 , :]

    mask = masked_crop[:,:,3]
    #solid_image = cv2.merge([mask,mask,mask])

    cropped_key_points = get_cropped_points(points, coords)

    resized = resize_(out_width=50,color_clusters=25,ksize=(10,10),image=masked_crop)

    limbs = get_limbs(cropped_key_points,resized)

    board = cv2.imread("sprites/white.png",cv2.IMREAD_UNCHANGED)

    resized_to_use = cv2.resize(resized, (150,450))

    dst = cv2.addWeighted(resized_to_use ,1.0,board ,1,0)

    imgs = []
    imgs.append(dst)
    angles = [7,-7,5,-5,10,-10,3,-3]
    for a in angles:
        i = rotate_image(dst,a)
        imgs.append(i)
    sprites = np.concatenate(imgs, axis=1)
    return {"sprites":sprites, "limbs":limbs}