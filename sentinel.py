import cv2
import time
import freenect
import curses
import random
import threading
import pyglet
import numpy as np
from PIL import Image

from launcher import *

threads = []

def exit_callback(dt):
    pyglet.app.exit() 

def play_audio(audio_type):
    if audio_type == 'fire':
        s = pyglet.media.load('audio/firing.mp3')
    elif audio_type == 'acquired person':
        s = pyglet.media.load('audio/there you are.mp3')
    elif audio_type == 'lost person':
        s = pyglet.media.load('audio/target lost.mp3')
    elif audio_type == 'searching':
        rand = random.random()
        if rand > 0.5:
            s = pyglet.media.load('audio/searching.mp3')
        else:
            s = pyglet.media.load('audio/are you still there.mp3')
    elif audio_type == 'shutdown':
        rand = random.random()
        if rand > 0.5:
            s = pyglet.media.load('audio/shutting down.mp3')
        else:
            s = pyglet.media.load('audio/goodnight.mp3')
    else:
        return
    s.play()
    pyglet.clock.schedule_once(exit_callback, s.duration)
    pyglet.app.run()
    s.delete()
    return

#def play_audio(audio_type):
#    t = threading.Thread(target=play_audio_subfunc, args = (audio_type,))
#    t.start()
   
class Sentinel(object):
    def __init__(self):
        #haar_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        haar_path = '/usr/local/Cellar/opencv/2.4.6.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(haar_path)

        print 'ZEROING LAUNCHER...'
        #self.launcher = Launcher()
        ## zero the launcher
        #self.launcher.run_command('down', 1000)
        #self.launcher.run_command('left', 6000)
        #self.launcher.run_command('right', 3000)
        #self.launcher.run_command('up', 500)
 
        print 'DONE'
        self.fx = 540
        self.fy = 540
        self.cx = 340
        self.cy = 240
        self.depth_constant = 0.0393701 # mm to inches

        freenect.sync_get_depth() 
        freenect.sync_get_video() 

    def display_face(self, face, image):
        # Faces -> [(x, y, w, h)]
        tmp = image.copy()
        tmp[:, :, 0] = 0
        tmp[:, :, 1] = 0
        tmp[:, :, 2] = image[:, :, 0]
        x, y, w, h = face
        cv2.rectangle(tmp, (x,y), (x+w,y+h), (255, 255, 255), 2)

        cv2.imshow('test', tmp)
        cv2.waitKey(10)

    def display_faces(self, faces, image):
        # Faces -> [(x, y, w, h)]
        tmp = image.copy()
        tmp[:, :, 0] = 0
        tmp[:, :, 1] = 0
        tmp[:, :, 2] = image[:, :, 0]
        for (x, y, w, h) in faces:
            cv2.rectangle(tmp, (x,y), (x+w,y+h), (0, 0, 255), 2)

        cv2.imshow('test', tmp)
        cv2.waitKey(10)

    def display_image(self, image):
        tmp = image.copy()
        tmp[:, :, 0] = 0
        tmp[:, :, 1] = 0
        tmp[:, :, 2] = image[:, :, 0]
        cv2.imshow('test', tmp)
        cv2.waitKey(10)

    def get_faces(self, image):
        faces = self.detector.detectMultiScale(image, 1.3, 3)
        return faces

    def update_face(self, prev_face_loc, depth_map):
        face_pix = self.get_face_pix(prev_face_loc)
        prev_face_depth = prev_face_loc[2]
        depth_map_cpy = np.copy(depth_map)
        depth_map_cpy[0:face_pix[1]-70, :] = 2047
        depth_map_cpy[face_pix[1]+70:, :] = 2047
        depth_map_cpy[:, 0:face_pix[0]-70] = 2047
        depth_map_cpy[:, face_pix[0]+70:] = 2047
        window = abs(depth_map_cpy - prev_face_depth)
        good_indices = np.where(window < 5)
        if not good_indices[0].any():
            return "Wrong"
        x = np.min(good_indices[1])
        y = np.min(good_indices[0])
        w = np.max(good_indices[1]) - x
        h = np.max(good_indices[0]) - y
        return [x, y, w, h]

    def get_face_depths(self, faces, depth_map):
        face_depths = []
        for face in faces:
            face_map = depth_map[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            med = np.median(face_map)
            bad_indices = abs(face_map - med) > 3
            face_map[bad_indices] = 0
            num_nonzero = np.count_nonzero(face_map)
            if num_nonzero != 0:
                face_depths.append(face_map.sum()/num_nonzero)
            else:
                face_depths.append(None)
            
        return face_depths

    def get_face_loc(self, face, depth):
        face_z = depth
        face_x = (face[0]-self.cx)*face_z/self.fx
        face_y = (face[1]-self.cy)*face_z/self.fy
        return (face_x, face_y, face_z)

    def get_face_pix(self, face_loc):
        x = self.fx*face_loc[0]/face_loc[2] + self.cx
        y = self.fy*face_loc[1]/face_loc[2] + self.cy
        return (x, y)

    def aim_follow(self, face_boxes, face_centers, face_depths, prev_face_loc, log_f):
        best_face = None
        best_dist = float('inf')
        best_face_idx = -1
        for face, depth, i in zip(face_centers, face_depths, range(len(face_centers))):
            face_loc = self.get_face_loc(face, depth)
            dist = np.sqrt((face_loc[0] - prev_face_loc[0])**2 + \
                           (face_loc[1] - prev_face_loc[1])**2 + \
                           (face_loc[2] - prev_face_loc[2])**2)
            if dist < best_dist:
                best_dist = dist
                best_face = face_loc
                best_face_idx = i

        if not best_face: # if we ended up trying to track a false positive
            return

        face = face_boxes[best_face_idx]
       
        lr_theta_1 = np.arcsin(best_face[0]/best_face[2])
        lr_theta_2 = np.arcsin(prev_face_loc[0]/prev_face_loc[2])
        lr_value = int((lr_theta_1 - lr_theta_2)*180/np.pi*6000/360)
        ud_theta_1 = np.arcsin(best_face[1]/best_face[2])
        ud_theta_2 = np.arcsin(prev_face_loc[1]/prev_face_loc[2])
        ud_value = int((ud_theta_1 - ud_theta_2)*180/np.pi*1000/30)
        
        #if (face[0] > 30 and face[1] > 30 and face[0]+face[2] < 450 and face[1]+face[3] < 610):
        #    if abs(lr_value) > 10: # noise in the measurements
        #        if lr_value < 0 and face[1]+face[3] < 640:
        #            self.launcher.run_command('left', -lr_value)
        #        elif lr_value > 0 and face[1] > 0:
        #            self.launcher.run_command('right', lr_value)
        #    if abs(ud_value) > 10: # noise in the measurements
        #        if ud_value < 0 and face[0]+face[2] < 480:
        #            self.launcher.run_command('up', -ud_value)
        #        elif ud_value > 0 and face[0] > 0:
        #            self.launcher.run_command('down', ud_value)
        
        face_center = face_centers[best_face_idx]
        log_f.write('{0}, {1}\n'.format(face_center[0], face_center[1]))
        log_f.write('{0}, {1}, {2}\n'.format(best_face[0], best_face[1], best_face[2]))
        log_f.write('{0}, {1}, {2}\n'.format(prev_face_loc[0], prev_face_loc[1], prev_face_loc[2]))
        log_f.write('{0} {1}\n\n'.format(lr_value, ud_value))

        return best_face_idx

    def guard(self, win):
    #def guard(self):
        win.nodelay(True) # make getkey() not wait
        stupid = 0
        print 'TRACKING...'
        faces = ()
        prev_face_loc = (1e-10, 1e-10, 20) # start pointing at origin
        nondetection_duration = 120
        detection_duration = 0
        f = open('log.txt', 'w+')
        while True:
            win.clear()
            win.addstr(0,0,str(stupid))
            stupid += 1
            try:
                key = win.getkey()
            except: # in no delay mode getkey raise and exeption if no key is press 
                key = None
            if key == "q": # exit on 'q'
                play_audio('shutdown')
                break
            if key == " ": # fire on space
                play_audio('fire')
                self.launcher.run_command('fire', 1)
            
            image_col = freenect.sync_get_video()[0]
            image = cv2.cvtColor(image_col, cv2.COLOR_BGR2GRAY)
            depth_map = freenect.sync_get_depth()[0]*self.depth_constant
            
            faces = self.get_faces(image)
           
            if faces != ():
                if nondetection_duration > 30 and detection_duration > 2:
                    play_audio('acquired person')
                    nondetection_duration = 0
                detection_duration += 1
            elif faces == () and nondetection_duration < 60 and prev_face_loc != (1e-10, 1e-10, 20): # fall back to depth tracking
                faces = [self.update_face(prev_face_loc, depth_map)]
                if faces == (): # this can fail too
                    nondetection_duration = 60
                detection_duration = 0
                nondetection_duration += 1
            elif nondetection_duration == 60:
                detection_duration = 0
                play_audio('lost person')
                nondetection_duration += 1
            elif nondetection_duration > 60 and nondetection_duration%120 == 0:
                play_audio('searching')
                nondetection_duration += 1

            if faces == () or faces == ["Wrong"]:
                self.display_image(image_col)
                continue
            face_centers = [(x[0]+x[2]/2, x[1]+x[3]/2) for x in faces]
            face_depths = self.get_face_depths(faces, depth_map)

            if face_depths == [None]:
                print 'Unable to acquire face depth, check thresholds'
                continue

            idx = self.aim_follow(faces, face_centers, face_depths, prev_face_loc, f)
            self.display_face(faces[idx], image_col)
            
            face = faces[idx]
            if (face[0] > 30 and face[1] > 30 and face[0]+face[2] < 450 and face[1]+face[3] < 610):
                prev_face_loc = self.get_face_loc(face_centers[idx], face_depths[idx])

        f.close()
    def activate(self):
        # Giant hacks to quit
        curses.wrapper(self.guard)
        #self.guard()

sentinel = Sentinel()
sentinel.activate()

