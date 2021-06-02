
from ia import OpenCVBase


class OpenPoseHand(OpenCVBase):

    def __init__(self):
        super().__init__("hand", "pose_deploy.prototxt", "pose_iter_102000.caffemodel", 21)