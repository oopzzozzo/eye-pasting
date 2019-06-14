import numpy as np
import cv2 as cv
import imutils
import dlib
import argparse

from utils import *
from blink import *
from match_face import *
from replace import replace

# a landmark predictor that should be downloaded from dlib
MARK_DETECT_PATH = 'shape_predictor_68_face_landmarks.dat'

# load detectors
face_detect = dlib.get_frontal_face_detector()
mark_detect = dlib.shape_predictor(MARK_DETECT_PATH)

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/report_out.jpg', help='output image path')
ap.add_argument('-a', action='store_true', help='should be specified when the two images are different in content' )
args = vars(ap.parse_args())


# mark all faces with 68 markers
def mark(img, marks):
    img = img.copy()
    for m68 in marks:
        for m in m68:
            cv.circle(img, tuple(m), 1, green)
    return img
            
def main():
    base = cv.imread(args['base'])
    material = cv.imread(args['eye'])

    # images are not taken sequentially
    if args['a']:
        matches = [(extreme_mark(base, face_detect, mark_detect, min), extreme_mark(material, face_detect, mark_detect, max))]
    else:
        bad = blink_faces(base, face_detect, mark_detect)
        matches = matchfaces(base, bad, material, face_detect, mark_detect)

    # outputs replacement marking of the two input images
    #cv.imwrite(args['base'].replace('in/','out/'), mark(base, [m[0] for m in matches]))
    #cv.imwrite(args['eye'].replace('in/','out/'), mark(material, [m[1] for m in matches]))

    mask_acc = np.zeros(base.shape, base.dtype)
    for match in matches:
        base, mask = replace(base, match[0], material, match[1])
        mask_acc += mask

    # outputs the mask used for replacement
    #cv.imwrite(args['base'].replace('in/','out/').replace('_c', '_mask'), mask_acc)
    cv.imwrite(args['out'], base)

    

if __name__ == '__main__':
    main()
