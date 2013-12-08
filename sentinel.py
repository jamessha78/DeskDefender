import cv2
import Image
import time
import freenect
import curses
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

    def display_face(self, face, image):
        # Faces -> [(x, y, w, h)]
        x, y, w, h = face
        cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)

        cv2.imshow('test', image)
        cv2.waitKey(10)

    def display_faces(self, faces, image):
        # Faces -> [(x, y, w, h)]
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)

        cv2.imshow('test', image)
        cv2.waitKey(10)

    def get_faces(self, image):
        faces = self.detector.detectMultiScale(image, 1.3, 3)
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

    def get_face_loc(self, face, depth):
        face_z = depth
        face_x = (face[0]-self.cx)*face_z/self.fx
        face_y = (face[1]-self.cx)*face_z/self.fy
        return (face_x, face_y, face_z)

    def aim_init(self, face_centers, face_depths):
        # just take first detection for now
        face_loc = self.get_face_loc(face_centers[0], face_depths[0])
        # TODO implement up-down angle
        # assume launcher (origin) is at bottom left hand corner of image
        # rotation of 100 ~ 5.5 inches at 60 inches
        lr_value = face_loc[0]/(5.5/60*face_loc[2])*100
        print lr_value
        self.launcher.run_command('right', lr_value)

    def aim_follow(self, face_centers, face_depths):
        best_face = None
        best_dist = float('inf')
        best_face_idx = -1
        for face, depth, i in zip(face_centers, face_depths, range(len(face_centers))):
            face_loc = self.get_face_loc(face, depth)
            dist = np.sqrt((face_loc[0] - self.prev_face_loc[0])**2 + \
                           (face_loc[1] - self.prev_face_loc[1])**2 + \
                           (face_loc[2] - self.prev_face_loc[2])**2)
            if dist < best_dist:
                best_dist = dist
                best_face = face_loc
                best_face_idx = i

        if not best_face: # if we ended up trying to track a false positive
            return
       
        lr_value = (best_face[0]-self.prev_face_loc[0])/(5.5/60*best_face[2])*100

        if abs(lr_value) > 10: # noise in the measurements
            print lr_value
            if lr_value < 0:
                self.launcher.run_command('left', -lr_value)
            else:
                self.launcher.run_command('right', lr_value)

        return best_face_idx

    def track(self, win):
        win.nodelay(True) # make getkey() not wait
        x = 0
        self.active_break = False
        while True:
            win.clear()
            win.addstr(0,0,str(x))
            x += 1
            try:
                key = win.getkey()
            except: # in no delay mode getkey raise and exeption if no key is press 
                key = None
            if key == "q": # of we got a q then break
                self.active_break = True
                break

            faces = ()
            for i in range(30): # try 30 times before giving up
                image = freenect.sync_get_video()[0]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                depth_map = freenect.sync_get_depth()[0]

                faces = self.get_faces(image)
                if faces != ():
                    break

            if faces == ():
                print 'LOST FACE... RESETTING...'
                self.launcher.run_command('zero', 0)
                break

            face_centers = [(x[0]+x[2]/2, x[1]+x[3]/2) for x in faces]
            face_depths = self.get_face_depths(faces, depth_map)

            idx = self.aim_follow(face_centers, face_depths)
            self.display_face(faces[idx], image)
            
            self.prev_face_loc = self.get_face_loc(face_centers[idx], face_depths[idx])


    def guard(self, win):
        win.nodelay(True) # make getkey() not wait
        x = 0
        print 'LOOKING FOR FACES...'
        faces = ()
        while faces == ():
            win.clear()
            win.addstr(0,0,str(x))
            x += 1
            try:
                key = win.getkey()
            except: # in no delay mode getkey raise and exeption if no key is press 
                key = None
            if key == "q": # of we got a space then break
                break

            image = freenect.sync_get_video()[0]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            depth_map = freenect.sync_get_depth()[0]
            self.display_faces(faces, image)

            faces = self.get_faces(image)
            if faces == ():
                continue

            print 'FOUND FACE...'
            face_centers = [(x[0]+x[2]/2, x[1]+x[3]/2) for x in faces]
            face_depths = self.get_face_depths(faces, depth_map)
            self.display_faces(faces, image)

            print 'AIMING...'
            self.aim_init(face_centers, face_depths)

            print 'TRACKING...'
            self.prev_face_loc = self.get_face_loc(face_centers[0], face_depths[0])
            curses.wrapper(self.track)
            faces = ()
            if self.active_break:
                break

    def activate(self):
        # Giant hacks to quit
        curses.wrapper(self.guard)

sentinel = Sentinel()
sentinel.activate()

