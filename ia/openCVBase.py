import os
import cv2
import sys
import json
import subprocess
from ia import BaseIA

class OpenCVBase(BaseIA):
    
    def __init__(self, name, protoFile, modelFile, size: int):
        super().__init__()
        self._model = None
        self._name = name
        self._protoFile = protoFile
        self._modelFile = modelFile
        self._size = size

    def getFile(self, fileName: str) -> str:
        if (fileName == None):
            return os.path.abspath("./ia-gen/openpose/models/pose/" + self._name)

        return os.path.abspath(os.path.join("./ia-gen/openpose/models/pose", self._name, fileName))

    def getModel(self):
        if (self._model == None):
            protoFile = self.getFile(self._protoFile)
            modelFile = self.getFile(self._modelFile)
            self._model = cv2.dnn.readNetFromCaffe(protoFile, modelFile)

        return self._model

    def getDataFile(self, word: str):
        output = os.path.abspath(os.path.join('./output_jsons', word))
        output = output.replace('?', '').replace('.', '')
        dataFile = os.path.join(output, 'dataset_opencv_' + self._name + '.json')
        return dataFile

    def validateWordIA(self, word: str):
        dataFile = self.getDataFile(word)
        return not os.path.isfile(dataFile)

    def processIAItem(self, word: str, video: str):
        print("Processar com body25: " + word)

        if (not self.validateWordIA(word)):
            return

        data = list()
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

            line = self.processFrameIA(width, height, blobImage)
            data.append(line)    
        

    def processFrameIA(self, width, height, blobImage):
        model = self.getModel()
        model.setInput(blobImage)
        outModel = model.forward()

        outSize = outModel.shape[1]
        outHeight = outModel.shape[2]
        outWidth = outModel.shape[3]
        line = list()
            
        for i in range(self._size):
            trustMap = outModel[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(trustMap)

            x = (width * point[0]) / outWidth
            y = (height * point[1] / outHeight)
            line.append(x)
            line.append(y)
            line.append(prob)
        
        return line

    def saveFileData(self, word: str, data: list):
        dataFile = self.getDataFile(word)
        dir = os.path.dirname(dataFile)

        if (not os.path.isdir(dir)):
            os.makedirs(dir)

        with open(dataFile, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)