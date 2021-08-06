import cv2
import json
import uuid
import subprocess
from ia import BaseIA
from ia import OpenCVBase

class MultipleOpenCVAI(BaseIA):

    def __init__(self, handAI: OpenCVBase):
        super().__init__()
        self._aiList = dict()
        self._handAI = handAI

    def addAI(self, ai: OpenCVBase):
        self._aiList[ai] = list()


    def processIAItem(self, word: str, video: str):
        print("Processar palavra: " + word)

        aiForWord = list(filter(lambda a: a.validateWordIA(word), self._aiList.keys()))

        capture = cv2.VideoCapture(video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = 0

        if (fps != 0):
            duration = frame_count/fps

        frame = 0
        width = 0
        height = 0
        listblobImages = list()

        while (cv2.waitKey(1) < 0):
            conected, video = capture.read()
            
            if not conected:
                cv2.waitKey()
                break

            width = video.shape[1]
            height = video.shape[0]

            listblobImages.append(video)

        altura_entrada = 360
        largura_entrada = int((altura_entrada / height) * width)

        blobImage = cv2.dnn.blobFromImages(listblobImages, 1.0 / 255, 
                                    (largura_entrada, altura_entrada), 
                                    (0, 0, 0), swapRB = False, crop = False)

        outModel = self._handAI.processBlobs(blobImage)
        handLine = self.getLine(outModel, width, height)

        self._handAI.saveFileData(word, handLine)

        for index in range(len(listblobImages)):
            self.criarImagem(listblobImages[index], handLine[index], index)

    def getLine(self, outModel, width, height):
        outSize = outModel.shape[1]
        outHeight = outModel.shape[2]
        outWidth = outModel.shape[3]
        itens = list()

        for l in range(outModel.shape[0]):
            line = list()

            for i in range(21):
                trustMap = outModel[l, i, :, :]
                minVal, prob, minLoc, point = cv2.minMaxLoc(trustMap)

                x = (width * point[0]) / outWidth
                y = (height * point[1] / outHeight)
                line.append(x)
                line.append(y)
                line.append(prob)

            itens.append(line)

        return itens

    def processIAItem2(self, word: str, video: str):
        print("Processar palavra: " + word)

        aiForWord = list(filter(lambda a: a.validateWordIA(word), self._aiList.keys()))

        capture = cv2.VideoCapture(video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = 0

        if (fps != 0):
            duration = frame_count/fps

        frame = 0

        while (cv2.waitKey(1) < 0):
            conected, video = capture.read()
            
            if not conected:
                cv2.waitKey()
                break

            width = video.shape[1]
            height = video.shape[0]
            time = 0

            if (fps != 0):
                time = frame / fps

            blobImage = cv2.dnn.blobFromImage(video, 1.0 / 255, 
                                    (width, height), 
                                    (0, 0, 0), swapRB = False, crop = False)

            id = str(uuid.uuid4())
            handLine = self._handAI.processFrameIA(blobImage, width, height)
            self._handAI.saveFileData(word, handLine)

            for aiWord in aiForWord:
                line = [
                    id,
                    fps,
                    frame_count,
                    duration,
                    width,
                    height,
                    frame,
                    time
                ]

                line = line + aiWord.processFrameIA(blobImage, width, height)
                line = line + handLine
                self._aiList[aiWord].append(line)

            frame = frame + 1

        for aiWord in aiForWord:
            aiWord.saveFileData(word, self._aiList[aiWord])


    def criarImagem(self, imagem, line, index):
        
        #hand
        pares_pontos = [
            [0, 1], [1, 2], [2, 3], [3, 4], 
            [0, 5], [5, 6], [6, 7], [7, 8], 
            [0, 9], [9, 10], [10, 11], [11, 12], 
            [0, 13], [13, 14], [14, 15], [15, 16], 
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]

        #MPII
        pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],[1,14],
               [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

        #COCO
        pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],[1,11],
               [11, 12], [12, 13], [1, 8], [8, 9], [9, 10], [0, 14], [14, 16], [0, 15], [15, 17]]

        


        cor_ponto, cor_linha = (255, 128, 0), (7, 62, 248)

        for par in pares_pontos:
            parteA = par[0]
            parteB = par[1]

            if (line[(parteA * 3) + 2] > 0.05 and line[(parteB * 3) + 2] > 0.05):
                pontoA = (int(line[parteA * 3]), int(line[(parteA * 3) + 1]))
                pontoB = (int(line[parteB * 3]), int(line[(parteB * 3) + 1]))

                cv2.line(imagem, pontoA, pontoB, cor_linha, 2)
                cv2.circle(imagem, pontoA, 3, cor_ponto, thickness = -1, lineType = cv2.LINE_AA)

        cv2.imwrite(r'D:\Code\TioRAC\libras-data-generation\output_jsons\ABACAXI\teste_' + str(index) + '.jpg', imagem)
