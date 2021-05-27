import os
import cv2
import uuid
import json


def processVideoData(listWords: list, listVideos: list):

    print("Processando Video Data")

    for index in range(listVideos.count()):
        word = listWords[index]
        video = listVideos[index]
        
        print("Processando vídeo data de " + word)

        videoFile = os.path.join(r'D:\Code\TioRAC\libras-data-generation\output_jsons', word)
        videoFile = videoFile.replace('?', '').replace('.', '')
        videoFile = os.path.join(videoFile, 'videoData.json')

        if (os.path.isfile(videoFile)):
            print("Arquivo de data de video já existe!")
        else:
            id = str(uuid.uuid4())
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = 0

            if (fps != 0):
                duration = frame_count/fps

            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            data = {
                "id": id,
                "fps": int(fps),
                "frame_count": frame_count,
                "duration": duration,
                "width": int(width),
                "height": int(height)
            }

            with open(videoFile, 'w') as outfile:
                json.dump(data, outfile)