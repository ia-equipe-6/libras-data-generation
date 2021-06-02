"""
Equipe 6

"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import videoDataGeneration as vdg
import datasetGenerate as dstGenerate

argparser = argparse.ArgumentParser(add_help=True)

def arg_str(string):
    return string

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

#argparser.add_argument('-json', dest="json", action='str', help='Arquivo JSON que vai carregar com dados dos vídeos')
argparser.add_argument('-json', dest="json", metavar='in-file', type=argparse.FileType('rt'), help='Arquivo JSON que vai carregar com dados dos vídeos')
argparser.add_argument('-f', dest="folder", type=dir_path, help='Pasta onde os arquivos estão localizados')
argparser.add_argument('-w', dest="word", type=arg_str, help='Chave da palavra dentro do json')
argparser.add_argument('-wf', dest="wordFile", type=arg_str, help='Chave do arquivo de vídeo de teste')
argparser.add_argument('-openPose', dest="openPose", action='store_true', help='OpenPose')
argparser.add_argument('-t', dest="thread", type=int, help='Quantidade de threads de processamento')
argparser.add_argument('-o', dest="output", metavar='out-file', type=argparse.FileType('wt'), help='Dados gerados')

args = argparser.parse_args()
inputDataset = pd.read_json(args.json)

if (args.word not in inputDataset):
    print("Coluna da palavra não encontrado")
    exit()

if (args.wordFile not in inputDataset):
    print("Coluna do nome do vídeo não encontrado")
    exit()

words = inputDataset[args.word]
files = inputDataset[args.wordFile]

files = files.apply(lambda f: f.replace(".swf", ".mp4"))
files = files.apply(lambda f: os.path.abspath(os.path.join(args.folder, f)))

from ia import MultipleOpenCVAI, OpenPoseBody25, OpenPoseCOCO, OpenPoseBody25B, OpenPoseMPII

#openpose = OpenPose()
multipleAI = MultipleOpenCVAI()
multipleAI.addAI(OpenPoseBody25())
multipleAI.addAI(OpenPoseCOCO())
multipleAI.addAI(OpenPoseBody25B())
multipleAI.addAI(OpenPoseMPII())
multipleAI.processIA(words, files)

vdg.processVideoData(words, files)
dstGenerate.generateDataset(words, files)

print("FIM")