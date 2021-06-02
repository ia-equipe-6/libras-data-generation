
from ia import OpenCVBase


class OpenPoseCOCO(OpenCVBase):

    def __init__(self):
        super().__init__("coco", "pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel", 18)