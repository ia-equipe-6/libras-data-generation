
from ia import OpenCVBase


class OpenPoseMPII(OpenCVBase):

    def __init__(self):
        super().__init__("mpi", "pose_deploy_linevec.prototxt", "pose_iter_160000.caffemodel", 18)