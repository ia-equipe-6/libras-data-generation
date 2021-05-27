import os
import sys
import subprocess
from ia import BaseIA

#.\bin\OpenPoseDemo.exe --video D:\Teste\LibrasVideos\ines\mp4\varios2Sm_Prog001.mp4 --write_json output_jsons/ --hand

class OpenPose(BaseIA):
    

    def processIA(self, listWords: list, listVideos: list):
        openPoseEXE = 'D:\\Code\\TioRAC\\libras-data-generation\\ia-gen\\openpose\\bin\\OpenPoseDemo.exe'
        cwd = r'D:\Code\TioRAC\libras-data-generation\ia-gen\openpose'
        print("Processando OpenPose")

        for index in range(listVideos.count()):
            word = listWords[index]
            video = listVideos[index]
            print("Processando OpenPose de " + word)

            output = os.path.join(r'D:\Code\TioRAC\libras-data-generation\output_jsons', word)
            output = output.replace('?', '').replace('.', '')

            if not os.path.exists(output):
                os.makedirs(output)

            if (os.listdir(output)):
                print("Palavra " + word + " ignorada, pasta cheia")
            else:
                args = [openPoseEXE, '--video', video, '--write_json', output, '--hand']
                process = subprocess.Popen(args, cwd=cwd, shell=True)
                process.wait()