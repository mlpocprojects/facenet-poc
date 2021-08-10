from keras import backend as K
import time
from multiprocessing.dummy import Pool
import requests
K.set_image_data_format('channels_first')
from cv2 import cv2
import glob
from collections import defaultdict
from numpy import genfromtxt
from fr_utils import *
from liveness_model import *
from inception_blocks_v2 import *
import win32com.client as wincl
from cv2.data import haarcascades
import pyttsx3

from time import time

PADDING = 50
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
start = time()
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

# load the liveness model
model = load_model()
end = time()
print(f'Total time taken for FRmodel and livenss model:{end - start}')
# initialisation
engine = pyttsx3.init()

# java server urls
java_url = 'http://43.231.127.150:7788/hello?names='

voices = engine.getProperty('voices')  # getting details of current voice
engine.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female


def triplet_loss(y_true, y_pred, alpha=0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    for i in range(maxFrames):
        pattern = '1' + '0' * (i + 1) + '1'
        if pattern in history:
            return True
    return False


# load the facnet-model weights
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)


def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database


def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_frontalface_default.xml'))
    face_detector = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_frontalface_alt.xml'))
    open_eyes_detector = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_eye_tree_eyeglasses.xml'))
    left_eye_detector = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_lefteye_2splits.xml'))
    right_eye_detector = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_righteye_2splits.xml'))

    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade, open_eyes_detector, left_eye_detector, right_eye_detector)

        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


def process_frame(img, frame, face_cascade, open_eyes_detector, left_eye_detector, right_eye_detector):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    eyes_detected = defaultdict(str)
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING

        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            identities.append(identity)

        face = frame[y:y + h, x:x + w]
        gray_face = gray[y:y + h, x:x + w]
        # Eyes detection
        # check first if eyes are open (with glasses taking into account)
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # if open_eyes_glasses detect eyes then they are open
        if len(open_eyes_glasses) == 2:
            eyes_detected[identity] += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # otherwise try detecting eyes using left and right_eye_detector
        # which can detect open and closed eyes
        else:
            # separate the face into left and right sides
            left_face = frame[y:y + h, x + int(w / 2):x + w]
            left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

            right_face = frame[y:y + h, x:x + int(w / 2)]
            right_face_gray = gray[y:y + h, x:x + int(w / 2)]

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            eye_status = '1'  # we suppose the eyes are open

            # For each eye check wether the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex, ey, ew, eh) in right_eye:
                color = (0, 255, 0)
                pred = predict(right_face[ey:ey + eh, ex:ex + ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(right_face, (ex, ey), (ex + ew, ey + eh), color, 2)
            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict(left_face[ey:ey + eh, ex:ex + ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(left_face, (ex, ey), (ex + ew, ey + eh), color, 2)
            eyes_detected[identity] += eye_status

        # Each time, we check if the person has blinked
        # If yes, we display its name
        if isBlinking(eyes_detected[identity], 3):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display name
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, identity, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.putText(frame, identity, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if identities:
        response = requests.get(java_url + ','.join(identities))
        print(response)
    #
    #     ready_to_detect_identity = False
    #     pool = Pool(processes=1)
    #     # We run this as a separate process so that the camera feedback does not freeze
    #     pool.apply_async(welcome_users, [identities])
    return img


def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    return who_is_it(part_image, database, FRmodel)


def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        # print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.52:
        return None
    else:
        return str(identity)


def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
    else:
        for identity_id in range(len(identities) - 1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'

    # windows10_voice_interface.Speak(welcome_message)
    engine.say(welcome_message)
    engine.runAndWait()

    # Allow the program to start detecting identities again
    ready_to_detect_identity = True


# Prepare the database
database = prepare_database()

# if __name__ == "__main__":
#
#     webcam_face_recognizer(database)
