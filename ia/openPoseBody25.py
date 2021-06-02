from ia import OpenCVBase


class OpenPoseBody25(OpenCVBase):

    def __init__(self):
        super().__init__("body_25", "pose_deploy.prototxt", "pose_iter_584000.caffemodel", 25)
