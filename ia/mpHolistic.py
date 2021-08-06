import os
import sys
import cv2
import uuid
import mediapipe as mp
from pathlib import Path
from ia import BaseIA


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


class MPHolistic(BaseIA):

    def processIAItem(self, word: str, video: str):

        data = list()
        capture = cv2.VideoCapture(video)

        output = os.path.abspath(os.path.join('./output_jsons', word))
        output = output.replace('?', '').replace('.', '')
        frame = 1

        with mp_holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
            while (cv2.waitKey(1) < 0):
                conected, image = capture.read()

                if not conected:
                    cv2.waitKey()
                    break

                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = 0

                if (fps != 0):
                    duration = frame_count / fps

                time = frame / fps

                image_width = image.shape[1]
                image_height = image.shape[0]
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
                results = holistic.process(image)

                if results.pose_landmarks:
                    print("Word: " + word + ", frame: " + str(frame) + ", time: " + str(time))

                line = [str(uuid.uuid4()),
                    word,
                    fps,
                    frame_count,
                    duration,
                    image_width,
                    image_height,
                    frame,
                    time
                ]

                line = self.createLine(results, line)
                data.append(line)
                self.createImage(results, image, output, video, frame)
                frame = frame + 1

        return data


               
    def createLine(self, results, line) -> list:
        line = self.createLinePose(results, line)
        line = self.createLineHand(results.left_hand_landmarks.landmark, line)
        line = self.createLineHand(results.right_hand_landmarks.landmark, line)
        return line

    def createLinePose(self, results, line: list) -> list:

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
        
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y)

        return line

    def createLineHand(self, landmark, line: list) -> list:
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].x)
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].y)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].y)

        return line

    def createImage(self, results, image, wordPath, video, frame):
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image=annotated_image, landmark_list=results.pose_landmarks, connections=mp_holistic.POSE_CONNECTIONS)

        videoPath = Path(video).stem
        imageFile = os.path.join(wordPath, videoPath)
        
        if (not os.path.isdir(imageFile)):
            os.mkdir(imageFile)
        
        imageFile = os.path.join(imageFile, "frame_" + str(frame) + ".jpg" )
        cv2.imwrite(imageFile, annotated_image) 

