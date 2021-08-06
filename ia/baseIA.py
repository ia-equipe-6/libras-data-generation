from abc import ABCMeta, abstractmethod
import pandas as pd

class BaseIA(object):
    __metaclass__ = ABCMeta


    def processIA(self, listWords: list, listVideos: list):
        print("Processando IA")
        data = list()

        for index in range(listVideos.count()):
            word = listWords[index]
            video = listVideos[index]
            data = data + self.processIAItem(word, video)

        self.saveData(data)

    @abstractmethod
    def processIAItem(self, word: str, video: str): pass

    def saveData(self, data):
        columns = self.getColumns()
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(r".\ten_words_dataset.csv", index = False)
        print(df)

    def getColumns(self):
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