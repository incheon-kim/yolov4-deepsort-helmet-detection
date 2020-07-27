from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from datetime import datetime
import matplotlib.pyplot as plt

FLAGS = {
    'HELMET_DRAW_ENABLED' : True,
    'SAVE_ON_NEW_HEAD' : True,
    'SHOW_ORIGINAL_IMAGE' : True,
    'SHOW_FPS' : True
}

def resizeDetections(original_image_shape, network_image_size, detections):
    resize_ratio = (original_image_shape[1]/network_image_size[0], original_image_shape[0]/network_image_size[1])
    resized_detections = []
    for detection in detections:
        resized_detections.append(
            (
                detection[0],
                detection[1],
                (
                detection[2][0] * resize_ratio[0],
                detection[2][1] * resize_ratio[1],
                detection[2][2] * resize_ratio[0],
                detection[2][3] * resize_ratio[1]
                )
            )
        )
    return resized_detections
    

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

def YOLO(imagepath):

    global metaMain, netMain, altNames
    configPath = "./configs/yolov4-helmet-detection.cfg"
    weightPath = "./configs/yolov4-helmet-detection.weights"
    metaPath = "./configs/yolov4-helmet-detection.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    # load video file / streams
    image = cv2.imread(imagepath)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    # network image size (416*416, ...)
    network_image_size = (darknet.network_width(netMain),
                          darknet.network_height(netMain))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,
                               network_image_size,
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    detections = darknet.detect_image(netMain,metaMain, darknet_image, thresh=0.25)
    detections = resizeDetections(image.shape, network_image_size, detections)
    
    detect_image = cvDrawBoxes(detections, image_rgb)
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)

    cv2.imshow("detected", detect_image)
    cv2.imwrite("output.jpg", detect_image)

    while True:
        # press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # file name goes here
    YOLO("example.png")
