from argparse import ArgumentParser
from input_feeder import InputFeeder
import os
from face_detection import FaceDetectorModel
from facial_landmarks_detection import FaceLandmarksDetector
from gaze_estimation import GazeEstimator
from head_pose_estimation import PoseDetector
from mouse_controller import MouseController
import cv2
import imutils
import math
import time
import logging
import numpy as np

def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")

    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of xml file of landmark regression model")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Head Pose Estimation model")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Gaze Estimation model")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Specify path of input Video file or cam for webcam")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flag from ff, fl, fh, fg like -flags ff fl(Space separated if multiple values)"
                             "ff for faceDetectionModel, fl for landmarkRegressionModel"
                             "fh for headPoseEstimationModel, fg for gazeEstimationModel")

    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify Device for inference"
                             "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    return parser

def init_model(args):
    global face_model, landmark_model, head_pose_model, gaze_model, mouse_controller
    device_name = args.device
    prob_threshold = args.prob_threshold

    # Initialize variables with the input arguments for easy access
    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }

    # Instantiate model
    face_model =FaceDetectorModel(model_path_dict['FaceDetectionModel'], device_name, threshold=prob_threshold)
    landmark_model = FaceLandmarksDetector(model_path_dict['LandmarkRegressionModel'], device_name, threshold=prob_threshold)
    head_pose_model = PoseDetector(model_path_dict['HeadPoseEstimationModel'], device_name, threshold=prob_threshold)
    gaze_model = GazeEstimator(model_path_dict['GazeEstimationModel'], device_name, threshold=prob_threshold)
    mouse_controller = MouseController('medium', 'fast')

    # Load Models
    face_model.load_model()
    landmark_model.load_model()
    head_pose_model.load_model()
    gaze_model.load_model()

    # Check extention of these unsupported layers
    face_model.check_model()
    landmark_model.check_model()
    head_pose_model.check_model()
    gaze_model.check_model()

def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')
    logging.basicConfig(filename='example.log',level=logging.ERROR)
    init_model(args)

    # Initialize variables with the input arguments for easy access
    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }

    preview_flags = args.previewFlags
    input_filename = args.input
    output_path = args.output_path
    prob_threshold = args.prob_threshold

    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    for model_path in list(model_path_dict.values()):
        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file" + str(model_path))
            exit(1)

    feeder.load_data()
    w = int(feeder.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feeder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feeder.cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (w, h), True)

    frame_count = 0
    for ret, frame in feeder.next_batch():
        if not ret:
            break
        frame_count += 1
        key = cv2.waitKey(60)

        try:
            cropped_image, face_cords = face_model.predict(frame, prob_threshold)
            
            if type(cropped_image) == int:
                print("Unable to detect the face")
                if key == 27:
                    break
                continue

            left_eye, right_eye, eye_cords = landmark_model.predict(cropped_image)
            pose_output = head_pose_model.predict(cropped_image)
            mouse_cord, gaze_vector = gaze_model.predict(left_eye, right_eye, pose_output)
        except Exception as e:
            print(str(e) + " for frame " + str(frame_count))
            continue

        image = cv2.resize(frame, (w, h))
        if not len(preview_flags) == 0:
            preview_frame = frame.copy()
            const = 10
            if 'ff' in preview_flags:
                if len(preview_flags) != 1:
                    preview_frame = cropped_image
                    cv2.rectangle(frame, (face_cords[0], face_cords[1]), (face_cords[2], face_cords[3]), (255, 0, 0), 3)

            if 'fl' in preview_flags:
                cv2.rectangle(cropped_image, (eye_cords[0][0]-const, eye_cords[0][1]-const), (eye_cords[0][2]+const, eye_cords[0][3]+const),
                      (0, 255, 0), 2)
                cv2.rectangle(cropped_image, (eye_cords[1][0]-const, eye_cords[1][1]-const), (eye_cords[1][2]+const, eye_cords[1][3]+const),
                      (0, 255, 0), 2)

            if 'fh' in preview_flags:
                cv2.putText(
                frame,
                "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                pose_output[0], pose_output[1], pose_output[2]),
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 255), 2)

            if 'fg' in preview_flags:
                x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
                le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                cv2.arrowedLine(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                cv2.arrowedLine(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                preview_frame[eye_cords[0][1]:eye_cords[0][3], eye_cords[0][0]:eye_cords[0][2]] = le
                preview_frame[eye_cords[1][1]:eye_cords[1][3], eye_cords[1][0]:eye_cords[1][2]] = re
            image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))

        cv2.imshow('preview', image)
        out_video.write(frame)

        if frame_count % 5 == 0:
            mouse_controller.move(mouse_cord[0], mouse_cord[1])

        if key == 27:
            break

    logger.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()

if __name__ == '__main__':
    main()
