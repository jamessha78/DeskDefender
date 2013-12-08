import cv2
import Image
import time
import freenect
import numpy as np

from launcher import *

class Sentinel(object):
    def __init__(self):
        #haar_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        haar_path = '/usr/local/Cellar/opencv/2.4.6.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(haar_path)

        print 'INITING LAUNCHER...'
        self.launcher = Launcher()
        # zero the launcher
        self.launcher.run_command('zero', 0)
        print 'DONE'
        self.fx = 540
        self.fy = 540
        self.cx = 0
        self.cy = 0
        self.depth_constant = 0.0393701 # mm to inches

        freenect.sync_get_depth() 
        freenect.sync_get_video() 

    def display_faces(self, faces, image):
        # Faces -> [(x, y, w, h)]
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)

        cv2.imshow('test', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_faces(self, image):
        faces = self.detector.detectMultiScale(image)
        return faces

    def get_face_depths(self, faces, depth_map):
        face_depths = []
        for face in faces:
            face_map = depth_map[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            num_nonzero = np.count_nonzero(face_map)
            if num_nonzero != 0:
                face_depths.append(face_map.sum()/num_nonzero*self.depth_constant)
            else:
                face_depths.append(None)
            
        return face_depths

    def aim_init(self, face_centers, face_depths):
        # Just take the first face for now
        face = face_centers[0]
        depth = face_depths[0]
        # assume launcher (origin) is at bottom left hand corner of image
        # rotation of 100 ~ 5.5 inches at 60 inches
        face_z = depth
        face_x = (face[0]-self.cx)*face_z/self.fx
        face_y = (face[1]-self.cx)*face_z/self.fy
        lr_value = face_x/(5.5/60*face_z)*100
        print lr_value
        self.launcher.run_command('right', lr_value)

    def aim_follow(self, face_centers, face_depths, prev_face_loc):
        best_face = None
        best_dist = float('inf')
        for i, face in enumerate(face_centers):
            depth = face_depths[i]
            face_z = depth
            face_x = (face[0]-self.cx)*face_z/self.fx
            face_y = (face[1]-self.cx)*face_z/self.fy
            dist = sqrt((face_x - prev_face_loc[0])**2 + \
                        (face_y - prev_face_loc[1])**2 + \
                        (face_z - prev_face_loc[2])**2)
            if dist < best_dist:
                best_dist = dist
                best_face = (face_x, face_y, face_z)
       
        lr_value = (best_face[0]-prev_face_loc[0])/(5.5/60*best_face[2])*100
        self.launcher.run_command('left', lr_value)

    def test(self):
        print 'GRABBING IMAGE AND DEPTH...'
        image = freenect.sync_get_video()[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth_map = freenect.sync_get_depth()[0]

        print 'DETECTING FACES...'
        faces = self.get_faces(image)
        face_centers = [(x[0]+x[2]/2, x[1]+x[3]/2) for x in faces]
        face_depths = self.get_face_depths(faces, depth_map)

        print 'AIMING...'
        self.aim_init(face_centers, face_depths)
sentinel = Sentinel()
sentinel.test()

