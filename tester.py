import glob
import pickle
import numpy as np

from PIL import Image
from FaceDetector import *
from generate_cropped_images import get_bounding_boxes
from utils import *

class TesterSlidingWindow(object):
    def __init__(self, thresholds):
        print "INITIALIZING..."
        cascade = pickle.load(open('cascade.pickle'))
        cascade.thresholds = [thresholds]
        face_detector = FaceDetector(cascade)
        self.classifier = face_detector
 
    def load_data(self, dir):
        print "LOADING DATA.."
        self.ground_truth = get_bounding_boxes('ground_truth.txt')
        self.test_images = {}
        file_list = glob.glob(dir + '*.gif')
        for file in file_list:
            # hacks to remove 'uncropped_images' from image name
            gt_key = file[17:]
            img = np.array(Image.open(file).convert('L'))
            self.test_images[gt_key] = img

    def test(self):
        print "TESTING..."
        true_positives = 0.0
        false_positives = 0.0
        gotten_positives = 0.0
        total_positives = 0.0
        for file in self.test_images.keys():
            print file
            img = self.test_images[file]
            detections = self.classifier.test(img)
            gts = self.ground_truth[file]
            for gt in gts:
                found = False
                total_positives += 1

                if detections == None:
                    continue
                
                for detection in detections:
                    # gt is (minx, miny, maxx, maxy)
                    # detection is (top, left, bottom, right) eg (miny, minx, maxy, maxx)
                    overlap_w = min(detection[3]-gt[0], gt[2]-detection[1])
                    overlap_h = min(detection[2]-gt[1], gt[3]-detection[0])
                    overlap_area = overlap_w*overlap_h
                    if overlap_w <= 0 or overlap_h <= 0:
                        continue
                    gt_w = gt[2]-gt[0]
                    gt_h = gt[3]-gt[1]
                    gt_area = gt_w*gt_h
                    det_w = detection[3]-detection[1]
                    det_h = detection[2]-detection[0]
                    det_area = det_w*det_h
                    if overlap_area >= 0.5*gt_area and overlap_area >= 0.5*det_area:
                        true_positives += 1
                        found = True
                    else:
                        false_positives += 1

                if found:
                    gotten_positives += 1
        
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives/(true_positives+false_positives)
        recall = gotten_positives/total_positives
        print 'TRUE POSITIVES', true_positives
        print 'FALSE POSITIVES', false_positives
        print 'FOUND POSITIVES', gotten_positives
        print 'TOTAL POSITIVES', total_positives
        print 'PRECISION', precision
        print 'RECALL', recall
        return precision, recall

class TesterSingleLabel(object):
    def __init__(self, thresholds):
        print "INITIALIZING..."
        cascade = pickle.load(open('cascade_pubfig.pickle'))
        cascade.thresholds = [thresholds]
        face_detector = FaceDetector(cascade)
        face_detector.window_scales = [1]
        self.classifier = face_detector
 
    def load_data(self, pos_dir, neg_dir):
        print "LOADING DATA.."
        #f = open('features_pubfig.pickle', 'r')
        #self.features = pickle.load(f)
        #self.pos_features = self.features[:400, :]
        #self.neg_features = self.features[-400:, :]
        #f.close()
        #f = open('labels_pubfig.pickle')
        #self.labels = pickle.load(f)
        #self.pos_labels = self.labels[:400]
        #self.neg_labels = self.labels[-400:]
        #f.close()

        f = open('features_cmu.pickle', 'r')
        self.features = pickle.load(f)
        self.pos_features = self.features[:169]
        self.neg_features = self.features[169:]
        f.close()
        f = open('labels_cmu.pickle', 'r')
        self.labels = pickle.load(f)
        self.pos_labels = self.labels[:169]
        print self.pos_labels
        self.neg_labels = self.labels[169:]
        f.close()

    def test(self):
        print "TESTING..."
        print 'pos', self.classifier.classifiers[2].score(self.pos_features, self.pos_labels)
        print 'neg', self.classifier.classifiers[2].score(self.neg_features, self.neg_labels)


#tester = TesterSlidingWindow(0.35)
#tester.load_data('uncropped_images/newtest/')
#precision, recall = tester.test()

f = open('precision_recall.txt', 'w+')
for i in range(0, 20):
    thresh = i/20.
    tester = TesterSlidingWindow(thresh)
    tester.load_data('uncropped_images/newtest/')
    precision, recall = tester.test()
    f.write('thresh: {0}, precision: {1}, recall: {2}\n'.format(thresh, precision, recall))
f.close()

#tester = TesterSingleLabel([0.5])
#pos_dir = '/Users/jsha/data/cropped_pubfig/'
#neg_dir = '/Users/jsha/data/negative_examples_pubfig/'
#tester.load_data(pos_dir, neg_dir)
#tester.test()
