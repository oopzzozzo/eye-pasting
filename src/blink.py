import cv2 as cv
import operator
from imutils import face_utils
from utils import get_eyes
from scipy.spatial import distance

ear_threshold = 0.25

avg_ear = lambda marks: sum(map(aspect_ratio, get_eyes(marks)))/2
def aspect_ratio(eye):
    vertical1 = distance.euclidean(eye[1], eye[5])
    vertical2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def get_marks_and_ear(img, rect, mark_detect):
    marks = face_utils.shape_to_np(mark_detect(img, rect))
    return marks, avg_ear(marks)

def blink_faces(img, face_detect, mark_detect):
    marks = [get_marks_and_ear(img, face, mark_detect) for face in face_detect(img, 1)]
    blinks = filter(lambda x: x[1] < ear_threshold, marks)
    return [x[0] for x in blinks]

def extreme_mark(img, face_detect, mark_detect, choose):
    marks = [get_marks_and_ear(img, face, mark_detect) for face in face_detect(img, 1)]
    return choose(marks, key=operator.itemgetter(1))[0]
