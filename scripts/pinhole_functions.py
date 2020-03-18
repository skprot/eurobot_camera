import cv2
from cv2 import aruco
import numpy as np

ARUCO_TABLE = np.array([[295, 1164], [250, 1047], [144, 1089], [188, 1206],
                        [1550, 800], [1450, 800], [1450, 700], [1550, 700]], dtype=np.float32) + 200 #2 markers with boat
MM_TO_PIX = 1 / 1.2254

def decreaseNoise(image, d=15, sigmaColor=100, sigmaSpace=100):
    filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return filtered_image

def crop(frame):
    frame_zero = np.zeros(frame.shape, dtype=np.uint8)
    frame_zero[455:, 415:2020] = frame[455:, 415:2020]
    return frame_zero

def orbFeatures(frame, path):
    img2 = cv2.imread(path)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img2, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print('\n matches:', matches)
    matching_result = cv2.drawMatches(frame, kp1, img2, kp2, matches[:10], None, flags=2)

    cv2.imwrite("../matches.jpg", matching_result)


def siftFeatures(frame, path):
    MIN_MATCH_COUNT = 20
    template = cv2.imread(path)

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    (keypoints1, descriptors1) = sift.detectAndCompute(frame_gray, None)
    (keypoints2, descriptors2) = sift.detectAndCompute(template_gray, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=200)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        img2_idx = m.trainIdx
        img1_idx = m.queryIdx
        pt1 = np.array(keypoints1[img1_idx].pt)
        pt2 = np.array(keypoints2[img2_idx].pt)
        dist = np.linalg.norm(pt1-pt2)
        if m.distance < 0.7 * n.distance and dist < 150:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_pts *= 2
        dst_pts *= 2

        M, mask = cv2.findHomography(src_pts, dst_pts)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    matches_img = cv2.drawMatches(frame_gray, keypoints1, template_gray, keypoints2, good, None, **draw_params)
    cv2.imwrite("../matches.jpg", matches_img)

    rospy.loginfo('Matches = ', len(good))

    return M


def rotate(arr):
    while not ((arr[0][0] > arr[1][0]) & (arr[3][1] < arr[0][1])):
        var = np.array(arr[3][:])

        for i in range(arr.shape[0]):
            if i == 3:
                arr[0][:] = var
                break
            arr[arr.shape[0] - i - 1][:] = arr[arr.shape[0] - i - 2][:]
    return arr


def orderCorners(arr):
    marker1 = np.copy(arr[:4][:])
    marker2 = np.copy(arr[4:8][:])


    rotate(marker1)
    rotate(marker2)

    if marker1[0][0] < marker2[0][0]:
        arr[:4][:] = np.copy(marker1)
        arr[4:8][:] = np.copy(marker2)
    else:
        arr[:4][:] = np.copy(marker2)
        arr[4:8][:] = np.copy(marker1)

    return arr


def perspectiveMatrix():
    perspective_points = ARUCO_TABLE * MM_TO_PIX
    print(perspective_points)
    return perspective_points


def center(frame, aruco_points):
    center_point = np.empty(2)
    center_point[0] = (aruco_points[3][0] - aruco_points[2][0]) / 2
    center_point[1] = aruco_points[3][1] + (aruco_points[0][1] - aruco_points[3][1]) / 2

    l = frame.shape[1] - 2 * center_point[1]
    frame = cv2.copyMakeBorder(frame, int(l), 0, 0, 0, borderType=cv2.BORDER_CONSTANT)

    for i in range(4):
        aruco_points[i][1] += l

    return frame, aruco_points


def homogeneousMatrix(frame):
    dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    corners, ids, _ = aruco.detectMarkers(frame, dict)
    try:
        if len(ids) != 2:
            return False
    except TypeError:
        return False

    for id in ids:
        if not (17 in id) | (42 in id):
            return False

    corner = np.array(corners).flatten()
    corner = np.reshape(corner, (-1, 2))

    orderCorners(corner)

    print(corner, '\n')
    return corner
