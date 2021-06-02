from ia import OpenCVBase


class OpenPoseBody25B(OpenCVBase):

    def __init__(self):
        super().__init__("body_25b", "pose_deploy.prototxt", "pose_iter_XXXXXX.caffemodel", 25)
