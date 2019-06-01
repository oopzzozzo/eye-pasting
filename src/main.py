import numpy as np
import cv2 as cv
import imutils
import dlib
import argparse
from imutils import face_utils
from scipy.spatial import distance


# a landmark predictor that should be downloaded from dlib
MARK_DETECT_PATH = 'shape_predictor_68_face_landmarks.dat'

# load detectors
face_detect = dlib.get_frontal_face_detector()
mark_detect = dlib.shape_predictor(MARK_DETECT_PATH)

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/out.jpg', help='output image path')
args = vars(ap.parse_args())

# helper fuctions to cast cv2 among dlib
rect2pts = lambda rect:((rect.left(), rect.top()), (rect.right(), rect.bottom()))
shape2pts = lambda shape: map(lambda i: (shape.part(i).x, shape.part(i).y), range(68))

# colors
green = (0, 255, 0)
red = (0, 0, 255)

def eye_aspect_ratio(eye):
    vertical1 = distance.euclidean(eye[1], eye[5])
    vertical2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

ear_theshold = 0.2

# mark all faces with 68 markers
def mark(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # argument "1" below upsamples the gray image to improve accuracy
    # lower -> not precise, higher -> time consuming
    faces = face_detect(gray_img, 1)
    for rect in faces:
        pts = rect2pts(rect)
        cv.rectangle(img, *pts, red, 1) # draw rectangle on img
        marks = mark_detect(gray_img, rect)
        for mark in shape2pts(marks):
            cv.circle(img, mark, 1, green) # draw circle on img
    return img

def detect_blink(img):
    (l_s_idx, l_e_idx) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_s_idx, r_e_idx) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detect(gray_img, 1)

    for rect in faces:
        marks = mark_detect(gray_img, rect)
        marks = face_utils.shape_to_np(marks)
        left_eye = marks[l_s_idx:l_e_idx]
        right_eye = marks[r_s_idx:r_e_idx]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        #print(avg_ear)

        # find blinked eye
        if avg_ear < ear_theshold:
            left_eye_hull = cv.convexHull(left_eye)
            right_eye_hull = cv.convexHull(right_eye)
            cv.drawContours(img, [left_eye_hull], -1, green, 1)
            cv.drawContours(img, [right_eye_hull], -1, green, 1)
    
    return img
            


def main():
    base = cv.imread(args['base'])
    #cv.imwrite('../out/close_out.jpg', mark(base))
    cv.imwrite('../out/close_blink.jpg', detect_blink(base))

    eye = cv.imread(args['eye'])
    #cv.imwrite('../out/open_out.jpg', mark(eye))
    cv.imwrite('../out/open_blink.jpg', detect_blink(eye))

    

if __name__ == '__main__':
    main()
