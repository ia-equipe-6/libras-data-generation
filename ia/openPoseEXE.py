import os
import sys
import subprocess
from ia import BaseIA

#.\bin\OpenPoseDemo.exe --video D:\Teste\LibrasVideos\ines\mp4\varios2Sm_Prog001.mp4 --write_json output_jsons/ --hand

class OpenPoseEXE(BaseIA):
    
    @property
    def openPoseEXE(self) -> str:
        return os.path.abspath('./ia-gen/openpose/bin/OpenPoseDemo.exe')

    @property
    def CWD(self) -> str:
        return os.path.abspath('./ia-gen/openpose')
                    

    def processIAItem(self, word: str, video: str):
        print("Processando OpenPose de " + word)

        output = os.path.aabspath(os.path.join('./output_jsons', word))
        output = output.replace('?', '').replace('.', '')

        if not os.path.exists(output):
            os.makedirs(output)

        if (os.listdir(output)):
            print("Palavra " + word + " ignorada, pasta cheia")
        else:
            args = [self.openPoseEXE, '--video', video, '--write_json', output, '--hand']
            process = subprocess.Popen(args, cwd=self.CWD, shell=True)
            process.wait()