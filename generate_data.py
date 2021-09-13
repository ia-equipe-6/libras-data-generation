"""
Gera dataset inicial a partir de vídeos.

    Giovanna Lima Marques 
    Ricardo Augusto Coelho 
    Tiago Goes Teles 
    Wellington de Jesus Albuquerque 

Busca vídeos na pasta 'videos', onde deve haver uma pasta por palavra podendo conter vários vídeos por plavra.
Todas as saídas são feita na pasta output, cada vídeo é gerado uma pasta com todos os frames gerados para verificação.
O dataset está em um csv dentro da pasta output.

Algumas referências:
    https://google.github.io/mediapipe/solutions/holistic.html
    https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing#scrollTo=BAivyQ_xOtFp

"""

import os
import cv2
import uuid
import pandas as pd
import mediapipe as mp
from pathlib import Path
import unicodedata

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
WITH_Z = True
ignored = 0

def process():
    """Processa todos os vídeos e salva o dataset"""
    data = list()
    words = [w for w in os.listdir("videos")] #Busca todas as palvras dentro da pasta 'videos'
    
    for word in words:
        wordFolder = os.path.join("videos", word)
        wordVideos = [v for v in os.listdir(wordFolder)]

        for wordVideo in wordVideos:
            videoFile = os.path.join("videos", word, wordVideo)
            data = data + processWord(word, videoFile)

    saveData(data)

def processWord(word, video) -> list:
    """Processa um único vídeo de uma única palavra"""
    global ignored
    data = list()
    capture = cv2.VideoCapture(video)

    output = os.path.abspath(os.path.join('./output', word))
    output = output.replace('?', '').replace('.', '')
    frame = 1
    wordId = str(uuid.uuid4()) #Gera um identificador único para o vídeo

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=2, min_detection_confidence=0.45) as holistic:
        while (cv2.waitKey(1) < 0): #Processa cada frame individualmente
            conected, image = capture.read() #Ler um frame

            if not conected:
                cv2.waitKey()
                break

            #Obtem dados do vídeo e frame
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = 0

            if (fps != 0):
                duration = frame_count / fps

            time = frame / fps

            image_width = image.shape[1]
            image_height = image.shape[0]

            #Prepara imagem para processo
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image) #Prever poses

            if results.pose_landmarks:
                print("Word: " + word + ", frame: " + str(frame) + ", time: " + str(time))

                #Cria uma linha do frame com dados do frame/vídeo
                line = [
                    wordId,
                    word,
                    fps,
                    frame_count,
                    duration,
                    image_width,
                    image_height,
                    frame,
                    time
                ]

                line = createLine(results, line)
                data.append(line)
                createImage(results, image, output, video, frame)
                frame = frame + 1
            else:
                #Não reconheceu nada da pose.
                print("IGNORADO Word: " + word + ", frame: " + str(frame) + ", time: " + str(time))
                #frame = frame + 1
                ignored += 1

    return data

def createLineEmptyValue(line: list, size = 40, value = 0) -> list:
    """Cria valores vazios por falta de reconhecimento de pose/mão"""
    for x in range(size):
        line.append(value)

    return line

def createLine(results, line) -> list:
    """Obtem todas as posições de pose e mão e adiciona na linha do dataset"""
    line = createLinePose(results, line)

    if (results.left_hand_landmarks != None):
        line = createLineHand(results.left_hand_landmarks.landmark, line)
    else:
        line = createLineEmptyValue(line, 40, 0)

    if (results.right_hand_landmarks != None):
        line = createLineHand(results.right_hand_landmarks.landmark, line)
    else:
        line = createLineEmptyValue(line, 40, 0)

    return line

def createLinePose(results, line: list) -> list:
    """Obtem os valores de poses do corpo"""

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

def createLineHand(landmark, line: list) -> list:
    """Obtem valores de poses da mão"""

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

def createImage(results, image, wordPath, video, frame):
    """Cria a imagem para verificação do processo"""
    annotated_image = image
    mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image=annotated_image, landmark_list=results.pose_landmarks, connections=mp_holistic.POSE_CONNECTIONS)

    videoPath = Path(video).stem
    imageFile = os.path.join(wordPath, videoPath)
    imageFile = unicodedata.normalize('NFD', imageFile)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")
    
    if (not os.path.isdir(imageFile)):
        os.makedirs(imageFile)
    
    imageFile = os.path.join(imageFile, "frame_" + str(frame) + ".jpg" )

    imageFile = unicodedata.normalize('NFD', imageFile)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")


    cv2.imwrite(imageFile, annotated_image)

def getColumns():
    """Obtem os nomes das colunas para salvar no CSV"""

    return [
        "ID",
        "WORD",
        "FPS",
        "FRAME_COUNT",
        "DURATION",
        "WIDTH",
        "HEIGHT",
        "FRAME",
        "TIME",
        "NOSE_X",
        "NOSE_Y",
        "LEFT_EYE_X",
        "LEFT_EYE_Y",
        "LEFT_EAR_X",
        "LEFT_EAR_Y",
        "RIGHT_EYE_X",
        "RIGHT_EYE_Y",
        "RIGHT_EAR_X",
        "RIGHT_EAR_Y",
        "MOUTH_LEFT_X",
        "MOUTH_LEFT_Y",
        "MOUTH_RIGHT_X",
        "MOUTH_RIGHT_Y",
        "LEFT_SHOULDER_X",
        "LEFT_SHOULDER_Y",
        "RIGHT_SHOULDER_X",
        "RIGHT_SHOULDER_Y",
        "LEFT_ELBOW_X",
        "LEFT_ELBOW_Y",
        "RIGHT_ELBOW_X",
        "RIGHT_ELBOW_Y",
        "LEFT_WRIST_X",
        "LEFT_WRIST_Y",
        "RIGHT_WRIST_X",
        "RIGHT_WRIST_Y",
        "LEFT_HIP_X",
        "LEFT_HIP_Y",
        "RIGHT_HIP_X",
        "RIGHT_HIP_Y",
        "LEFT_HAND_THUMB_CMC_X",
        "LEFT_HAND_THUMB_CMC_Y",
        "LEFT_HAND_THUMB_MCP_X",
        "LEFT_HAND_THUMB_MCP_Y",
        "LEFT_HAND_THUMB_IP_X",
        "LEFT_HAND_THUMB_IP_Y",
        "LEFT_HAND_THUMB_TIP_X",
        "LEFT_HAND_THUMB_TIP_Y",
        "LEFT_HAND_INDEX_FINGER_MCP_X",
        "LEFT_HAND_INDEX_FINGER_MCP_Y",
        "LEFT_HAND_INDEX_FINGER_PIP_X",
        "LEFT_HAND_INDEX_FINGER_PIP_Y",
        "LEFT_HAND_INDEX_FINGER_DIP_X",
        "LEFT_HAND_INDEX_FINGER_DIP_Y",
        "LEFT_HAND_INDEX_FINGER_TIP_X",
        "LEFT_HAND_INDEX_FINGER_TIP_Y",
        "LEFT_HAND_MIDDLE_FINGER_MCP_X",
        "LEFT_HAND_MIDDLE_FINGER_MCP_Y",
        "LEFT_HAND_MIDDLE_FINGER_PIP_X",
        "LEFT_HAND_MIDDLE_FINGER_PIP_Y",
        "LEFT_HAND_MIDDLE_FINGER_DIP_X",
        "LEFT_HAND_MIDDLE_FINGER_DIP_Y",
        "LEFT_HAND_MIDDLE_FINGER_TIP_X",
        "LEFT_HAND_MIDDLE_FINGER_TIP_Y",
        "LEFT_HAND_RING_FINGER_MCP_X",
        "LEFT_HAND_RING_FINGER_MCP_Y",
        "LEFT_HAND_RING_FINGER_PIP_X",
        "LEFT_HAND_RING_FINGER_PIP_Y",
        "LEFT_HAND_RING_FINGER_DIP_X",
        "LEFT_HAND_RING_FINGER_DIP_Y",
        "LEFT_HAND_RING_FINGER_TIP_X",
        "LEFT_HAND_RING_FINGER_TIP_Y",
        "LEFT_HAND_PINKY_MCP_X",
        "LEFT_HAND_PINKY_MCP_Y",
        "LEFT_HAND_PINKY_PIP_X",
        "LEFT_HAND_PINKY_PIP_Y",
        "LEFT_HAND_PINKY_DIP_X",
        "LEFT_HAND_PINKY_DIP_Y",
        "LEFT_HAND_PINKY_TIP_X",
        "LEFT_HAND_PINKY_TIP_Y",
        "RIGHT_HAND_THUMB_CMC_X",
        "RIGHT_HAND_THUMB_CMC_Y",
        "RIGHT_HAND_THUMB_MCP_X",
        "RIGHT_HAND_THUMB_MCP_Y",
        "RIGHT_HAND_THUMB_IP_X",
        "RIGHT_HAND_THUMB_IP_Y",
        "RIGHT_HAND_THUMB_TIP_X",
        "RIGHT_HAND_THUMB_TIP_Y",
        "RIGHT_HAND_INDEX_FINGER_MCP_X",
        "RIGHT_HAND_INDEX_FINGER_MCP_Y",
        "RIGHT_HAND_INDEX_FINGER_PIP_X",
        "RIGHT_HAND_INDEX_FINGER_PIP_Y",
        "RIGHT_HAND_INDEX_FINGER_DIP_X",
        "RIGHT_HAND_INDEX_FINGER_DIP_Y",
        "RIGHT_HAND_INDEX_FINGER_TIP_X",
        "RIGHT_HAND_INDEX_FINGER_TIP_Y",
        "RIGHT_HAND_MIDDLE_FINGER_MCP_X",
        "RIGHT_HAND_MIDDLE_FINGER_MCP_Y",
        "RIGHT_HAND_MIDDLE_FINGER_PIP_X",
        "RIGHT_HAND_MIDDLE_FINGER_PIP_Y",
        "RIGHT_HAND_MIDDLE_FINGER_DIP_X",
        "RIGHT_HAND_MIDDLE_FINGER_DIP_Y",
        "RIGHT_HAND_MIDDLE_FINGER_TIP_X",
        "RIGHT_HAND_MIDDLE_FINGER_TIP_Y",
        "RIGHT_HAND_RING_FINGER_MCP_X",
        "RIGHT_HAND_RING_FINGER_MCP_Y",
        "RIGHT_HAND_RING_FINGER_PIP_X",
        "RIGHT_HAND_RING_FINGER_PIP_Y",
        "RIGHT_HAND_RING_FINGER_DIP_X",
        "RIGHT_HAND_RING_FINGER_DIP_Y",
        "RIGHT_HAND_RING_FINGER_TIP_X",
        "RIGHT_HAND_RING_FINGER_TIP_Y",
        "RIGHT_HAND_PINKY_MCP_X",
        "RIGHT_HAND_PINKY_MCP_Y",
        "RIGHT_HAND_PINKY_PIP_X",
        "RIGHT_HAND_PINKY_PIP_Y",
        "RIGHT_HAND_PINKY_DIP_X",
        "RIGHT_HAND_PINKY_DIP_Y",
        "RIGHT_HAND_PINKY_TIP_X",
        "RIGHT_HAND_PINKY_TIP_Y",
    ]

def saveData(data):
    """Salva o dataset gerado"""
    columns = getColumns()
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(r"./output/words_dataset.csv", index = False)
    print(df)
    print(ignored)

#Inicia o processo, pelo método process:
process()