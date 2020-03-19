#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import detector
import cv2
import numpy as np
import subprocess
import pinhole_functions
import time
import color_detection


class CameraNode:
    def __init__(self):
        self.node = rospy.init_node('camera', anonymous=True)
        self.K = np.asarray(rospy.get_param("K"))
        self.D = np.asarray(rospy.get_param("D"))
        self.DIM = tuple(rospy.get_param("DIM"))
        self.projection_matrix = np.asarray(rospy.get_param("PROJECTION_MATRIX"))
        self.template_path = rospy.get_param("TEMPLATE_PATH")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.DIM[0])
        self.cap.set(4, self.DIM[1])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM,
                                                                   cv2.CV_16SC2)

        self.model_cfg_path = rospy.get_param("MODEL_CONFIG_PATH")
        self.model_weights_path = rospy.get_param("MODEL_WEIGHTS_PATH")
        self.classes = rospy.get_param("CLASSES")
        self.cup_detector = detector.Detect(self.model_cfg_path, self.model_weights_path, self.classes, confidence=0.4)

        self.seq_publisher = rospy.Publisher('/sequence', String, queue_size=1)
        self.compas_publisher = rospy.Publisher('/wind_direction', String, queue_size=1)
        self.cups_publisher = rospy.Publisher('/field_cups', String, queue_size=1)
        self.reef_publisher = rospy.Publisher('/reef_presence', String, queue_size=1)
        self.field_publisher = rospy.Publisher('/field_presence', String, queue_size=1)

        rospy.Subscriber('/main_robot/stm/start_status', String, self.start_status_callback_main, queue_size=1)
        rospy.Subscriber('/secondary_robot/stm/start_status', String, self.start_status_callback_secondary, queue_size=1)

        self.timer = -1
        self.seq = ""
        self.compas = ""
        self.start_status = ""
        self.matrix_projection = 0
        self.crop_mask = 0

        self.find_feature_matrix()

        rospy.logwarn("INITIALIZATION COMPLETED")
        rospy.logwarn("CAMERA CYCLE STARTED")
        self.run()

    def start_status_callback_main(self, data):
        self.start_status = data.data

    def start_status_callback_secondary(self, data):
        self.start_status = data.data

    def run(self):
        start_flag = False
        while not rospy.is_shutdown():
            if self.start_status == "1" and not start_flag:
                self.timer = time.time()
                rospy.logwarn('MATCH STARTED')
                start_flag = True

            ret, frame = self.cap.read()
            undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            undistorted = cv2.bitwise_and(undistorted, undistorted, mask=self.crop_mask)
            # TODO first shot?
            output = self.cup_detector.detect(undistorted)
            # TODO crop function

            if start_flag and (time.time() - self.timer) > 0:

                if self.seq == "":
                    # TODO REMAKE AFTER RETRAINIG
                    seq_frame = cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740))
                    #_, colors, self.seq = color_detection.findColors(seq_frame
                    _, self.seq = color_detection.findColorsHSV(seq_frame)


                if self.compas == "" and (time.time() - self.timer) > 30:
                    compas_frame = cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740))
                    self.compas = color_detection.findCompas(compas_frame)

                print('Timer: ', time.time() - self.timer)
                #self.cups_publisher.publish(output) test smth
                #rospy.logwarn(output) test smth
                cv2.imshow('cups', output)
                cv2.waitKey(1)

                self.compas_publisher.publish(self.compas)
                rospy.logwarn(self.compas)
                self.seq_publisher.publish(self.seq)
                rospy.logwarn(self.seq)

            if start_flag and (time.time() - self.timer) > 120:
                rospy.logwarn("MATCH ENDED")
                return 0

    def find_feature_matrix(self):
        frame_num = 0

        while frame_num < 10:
            ret, frame = self.cap.read()
            frame_num += 1

        undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        seq_frame = cv2.warpPerspective(undistorted, self.projection_matrix, (3000, 2000))
        matrix_feature = pinhole_functions.siftFeatures(seq_frame, self.template_path)
        self.matrix_projection = np.dot(matrix_feature, self.projection_matrix)
        croped_img = pinhole_functions.crop(cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740)))
        self.crop_mask = cv2.warpPerspective(croped_img, np.linalg.inv(self.matrix_projection), self.DIM)
        self.crop_mask = cv2.cvtColor(self.crop_mask, cv2.COLOR_BGR2GRAY)
        _, self.crop_mask = cv2.threshold(self.crop_mask, 1, 1, cv2.THRESH_BINARY)

if __name__ == '__main__':

    try:
        CameraNode()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
