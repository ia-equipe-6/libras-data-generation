import cv2
import json
import subprocess
from ia import BaseIA
from ia import OpenCVBase

class MultipleOpenCVAI(BaseIA):

    def __init__(self):
        super().__init__()
        self._aiList = dict()

    def addAI(self, ai: OpenCVBase):
        self._aiList[ai] = list()

    def processIAItem(self, word: str, video: str):
        print("Processar palavra: " + word)

        aiForWord = list(filter(lambda a: a.validateWordIA(word), self._aiList.keys()))

        capture = cv2.VideoCapture(video)

        while (cv2.waitKey(1) < 0):
            conected, video = capture.read()
            
            if not conected:
                cv2.waitKey()
                break

            width = video.shape[1]
            height = video.shape[0]

            blobImage = cv2.dnn.blobFromImage(video, 1.0 / 255, 
                                    (width, height), 
                                    (0, 0, 0), swapRB = False, crop = False)

            for aiWord in aiForWord:
                line = aiWord.processFrameIA(width, height, blobImage)
                self._aiList[aiWord].append(line)

        for aiWord in aiForWord:
            aiWord.saveFileData(word, self._aiList[aiWord])