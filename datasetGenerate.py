import os
import json
import pandas as pd

def generateDataset(listWords: list, listVideos: list):
    print("Gerando dataset")

    data = list()

    for index in range(listVideos.count()):
        linesWords = generateLinesWord(index, listWords, listVideos)
        data = data + linesWords
    
    columns = generateColumnsName()

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(r".\ines_dataset.csv", index = False)
    print(df)

def loadJsonData(file):
    if (os.path.isfile(file)):
        f = open(file,)
        jsonData = json.load(f)
        f.close()
        return jsonData

    return {}

def generateColumnsName():
    columns = [
        "id",
        "word",
        "fps",
        "frame_count",
        "duration",
        "width",
        "height",
        "frame",
        "time"
    ]

    for pos in range(0, 75):
        columns.append("body_" + str(pos))

    for pos in range(0, 63):
        columns.append("left_hand_" + str(pos))

    for pos in range(0, 63):
        columns.append("right_hand_" + str(pos))

    return columns

def generateLinesWord(index: int, listWords: list, listVideos: list):
    word = listWords[index]
    video = listVideos[index]
    print("Processando linha de " + word)

    videoFolder = os.path.join(r'D:\Code\TioRAC\libras-data-generation\output_jsons', word)
    videoFolder = videoFolder.replace('?', '').replace('.', '')

    videoDataFile = os.path.join(videoFolder, 'videoData.json')
    videoData = loadJsonData(videoDataFile)

    videoFiles = os.listdir(videoFolder)
    linesData = list()

    for videoFrameData in videoFiles:
        line = generateLine(word, videoData, videoFolder, videoFiles, videoFrameData)

        if (line != None):
            linesData.append(line)

    return linesData

def generateLine(word, videoData, videoFolder, videoFiles, videoFrameData):
    if (videoFrameData.endswith('_keypoints.json')):
        #print("Criando dados de: " + videoFrameData)
        filename = os.path.basename(videoFrameData)
        (file, ext) = os.path.splitext(filename)
        nameSplit = file.split('_')
        frameData = loadJsonData(os.path.join(videoFolder, videoFrameData))

        if ("people" in frameData
            and len(frameData["people"]) >= 1
            and "pose_keypoints_2d" in frameData["people"][0]
            and len(frameData["people"][0]["pose_keypoints_2d"]) == 75
            and "hand_left_keypoints_2d" in frameData["people"][0]
            and len(frameData["people"][0]["hand_left_keypoints_2d"]) == 63
            and "hand_right_keypoints_2d" in frameData["people"][0]
            and len(frameData["people"][0]["hand_right_keypoints_2d"]) == 63):

            peopleData = frameData["people"][0] 
            bodyData = peopleData["pose_keypoints_2d"]
            leftHandData = peopleData["hand_left_keypoints_2d"]
            rightHandData = peopleData["hand_right_keypoints_2d"]

            frame = int(nameSplit[-2])
            time = frame / int(videoData["fps"])

            line = [
                videoData["id"],
                word,
                videoData["fps"],
                videoData["frame_count"],
                videoData["duration"],
                videoData["width"],
                videoData["height"],
                frame,
                time
            ]

            for pos in bodyData:
                line.append(pos)

            for pos in leftHandData:
                line.append(pos)

            for pos in rightHandData:
                line.append(pos)

            return line
        else:
            print("Dados inesperados...")

    else:
        print("Não é um keypoint")

    return None