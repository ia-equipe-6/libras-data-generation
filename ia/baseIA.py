from abc import ABCMeta, abstractmethod


class BaseIA(object):
    __metaclass__ = ABCMeta


    @abstractmethod
    def processIA(self, listWords: list, listVideos: list):
        pass