import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def findColors(frame):
    colors = {
        "Green" : 0,
        "Red" : 1
    }

    cups_img = frame[1662:1710, 430:830]
    Z = cups_img.reshape((-1, 3))
    Z_fl = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z_fl, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(cups_img.shape)
    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    av_cups_img = []
    for i in range(res2.shape[1]):
        temp = 0
        for j in range(res2.shape[0]):
            temp += res2[j][i]
        temp /= res2.shape[0]
        av_cups_img.append(temp)

    avg = np.mean(av_cups_img)
    segment_length = cups_img.shape[1] / 5
    segment_list = []

    j = 1
    av_segment = 0
    for i in range(len(av_cups_img)):
        av_segment += av_cups_img[i]
        if i > j * segment_length - 2:
            av_segment /= segment_length
            segment_list.append(av_segment)
            j += 1
            av_segment = 0

    color_sequence = []
    color_str = ""
    for i in range(len(segment_list)):
        if segment_list[i] < avg:
            color_sequence.append(colors["Green"])
            color_str += "G"
        else:
            color_sequence.append(colors["Red"])
            color_str += "R"

    return res2, color_sequence, color_str

def findCompas(frame):
    compas_img = frame[1608:1620, 1200:1260]
    compas_img = cv2.cvtColor(compas_img, cv2.COLOR_BGR2GRAY)
    ret, tresh = cv2.threshold(compas_img, 127, 255, cv2.THRESH_BINARY)
    avg = cv2.mean(tresh)
    avg = avg[0]

    if avg == 255:
        return "South"
    elif avg == 0:
        return "North"
    else:
        return "fucked"
