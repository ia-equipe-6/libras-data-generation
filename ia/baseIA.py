from abc import ABCMeta, abstractmethod


class BaseIA(object):
    __metaclass__ = ABCMeta


    def processIA(self, listWords: list, listVideos: list):
        print("Processando IA")

        for index in range(listVideos.count()):
            word = listWords[index]
            video = listVideos[index]
            self.processIAItem(word, video)

    @abstractmethod
    def processIAItem(self, word: str, video: str): pass