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
    https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing#scrollTo=BAivyQOtFp

"""

import os
import cv2
import uuid
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
import unicodedata

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
WITH_Z = False
ignored = 0

DESIRED_HEIGHT = 200
DESIRED_WIDTH = 200

def process():
    """Processa todos os vídeos e salva o dataset"""
    data = list()
    words = [w for w in os.listdir("videos")] #Busca todas as palvras dentro da pasta 'videos'
    words.sort()
    
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

    if (word == "0"):
        word = "None"

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

            h, w = image.shape[:2]

            if h < w:
                image = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
            else:
                image = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

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

            if (not results.pose_landmarks):
                #Não reconheceu bem, voltar as cores para RGB para ver se reconhece melhor
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

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
                createImage(results, image, output, video + "FALHA.mp4", frame)
                #frame = frame + 1
                ignored += 1
                exit()

    return data

def createLineEmptyValue(line: list, size = 40, value = 0) -> list:
    """Cria valores vazios por falta de reconhecimento de pose/mão"""
    for x in range(size):
        line.append(value)

    return line

def createLine(results, line) -> list:
    """Obtem todas as posições de pose e mão e adiciona na linha do dataset"""
    line = createLinePose(results, line)
    handSize = 40
    if (WITH_Z):
        handSize = 60

    if (results.left_hand_landmarks != None):
        line = createLineHand(results.left_hand_landmarks.landmark, line)
    else:
        line = createLineEmptyValue(line, handSize, 0)

    if (results.right_hand_landmarks != None):
        line = createLineHand(results.right_hand_landmarks.landmark, line)
    else:
        line = createLineEmptyValue(line, handSize, 0)

    return line

def createLinePose(results, line: list) -> list:
    """Obtem os valores de poses do corpo"""

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z)
    
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z)

    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x)
    line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y)
    if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z)

    return line

def createLineHand(landmark, line: list) -> list:
    """Obtem valores de poses da mão"""

    #line.append(landmark[mp_holistic.HandLandmark.WRIST].x)
    #line.append(landmark[mp_holistic.HandLandmark.WRIST].y)

    line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].x)
    line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].z)

    line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].x)
    line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].z)

    line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].x)
    line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].z)

    line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
    line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].z)

    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)
    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z)

    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)
    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z)

    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)
    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z)

    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
    line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z)

    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)
    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z)

    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)
    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z)

    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)
    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z)

    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
    line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z)

    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)
    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z)

    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)
    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z)

    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)
    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z)

    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
    line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z)

    line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].x)
    line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].z)

    line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].x)
    line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].z)

    line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].x)
    line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].z)

    line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
    line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].y)
    if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].z)

    return line

def createImage(results, image, wordPath, video, frame):
    """Cria a imagem para verificação do processo"""
    h, w = image.shape[:2]

    annotated_image = np.zeros((DESIRED_WIDTH, DESIRED_HEIGHT, 3), np.uint8)
    annotated_image[:] = (255, 255, 255)
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
    
    imageFile = os.path.join(imageFile, "frame_" + str(frame) + ".png" )

    imageFile = unicodedata.normalize('NFD', imageFile)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")


    cv2.imwrite(imageFile, annotated_image)

def generateColumns(columns: list, withZ: bool) -> list:
    allColumns = list()

    for column in columns:
        allColumns.append(column + "_X")
        allColumns.append(column + "_Y")
        if withZ: allColumns.append(column + "_Z")

    return allColumns

def getColumns():
    """Obtem os nomes das colunas para salvar no CSV"""

    columns = [
        "ID",
        "WORD",
        "FPS",
        "FRAME_COUNT",
        "DURATION",
        "WIDTH",
        "HEIGHT",
        "FRAME",
        "TIME"]

    toGenerate = [
        "NOSE",
        "LEFT_EYE",
        "LEFT_EAR",
        "RIGHT_EYE",
        "RIGHT_EAR",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_HAND_THUMB_CMC",
        "LEFT_HAND_THUMB_MCP",
        "LEFT_HAND_THUMB_IP",
        "LEFT_HAND_THUMB_TIP",
        "LEFT_HAND_INDEX_FINGER_MCP",
        "LEFT_HAND_INDEX_FINGER_PIP",
        "LEFT_HAND_INDEX_FINGER_DIP",
        "LEFT_HAND_INDEX_FINGER_TIP",
        "LEFT_HAND_MIDDLE_FINGER_MCP",
        "LEFT_HAND_MIDDLE_FINGER_PIP",
        "LEFT_HAND_MIDDLE_FINGER_DIP",
        "LEFT_HAND_MIDDLE_FINGER_TIP",
        "LEFT_HAND_RING_FINGER_MCP",
        "LEFT_HAND_RING_FINGER_PIP",
        "LEFT_HAND_RING_FINGER_DIP",
        "LEFT_HAND_RING_FINGER_TIP",
        "LEFT_HAND_PINKY_MCP",
        "LEFT_HAND_PINKY_PIP",
        "LEFT_HAND_PINKY_DIP",
        "LEFT_HAND_PINKY_TIP",
        "RIGHT_HAND_THUMB_CMC",
        "RIGHT_HAND_THUMB_MCP",
        "RIGHT_HAND_THUMB_IP",
        "RIGHT_HAND_THUMB_TIP",
        "RIGHT_HAND_INDEX_FINGER_MCP",
        "RIGHT_HAND_INDEX_FINGER_PIP",
        "RIGHT_HAND_INDEX_FINGER_DIP",
        "RIGHT_HAND_INDEX_FINGER_TIP",
        "RIGHT_HAND_MIDDLE_FINGER_MCP",
        "RIGHT_HAND_MIDDLE_FINGER_PIP",
        "RIGHT_HAND_MIDDLE_FINGER_DIP",
        "RIGHT_HAND_MIDDLE_FINGER_TIP",
        "RIGHT_HAND_RING_FINGER_MCP",
        "RIGHT_HAND_RING_FINGER_PIP",
        "RIGHT_HAND_RING_FINGER_DIP",
        "RIGHT_HAND_RING_FINGER_TIP",
        "RIGHT_HAND_PINKY_MCP",
        "RIGHT_HAND_PINKY_PIP",
        "RIGHT_HAND_PINKY_DIP",
        "RIGHT_HAND_PINKY_TIP",
    ]

    columns = columns + generateColumns(toGenerate, WITH_Z)

    return columns
        

def saveData(data):
    """Salva o dataset gerado"""
    columns = getColumns()
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(r"./output/words_dataset.csv", index = False)
    print(df)
    print("Frames ignorados: " + str(ignored))

#Inicia o processo, pelo método process:
process()