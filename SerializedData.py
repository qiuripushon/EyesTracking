from tkinter import ROUND
import numpy as np
import json


def loadSerializedData(filename : str, roundNum : int):
    f = open(filename, "r")
    data = f.readlines()
    f.close()

    target = [ ]
    gazeInfo = [ [] for _ in range(roundNum)]

    idx = 0
    for i in range(roundNum):
        # load target pos
        item = json.loads(data[idx])
        target.append( np.array([ item["target"]["x"], item["target"]["y"], item["target"]["z"] ]))
        idx += 1
        # load gaze etc.
        while idx < len(data):
            item = json.loads(data[idx])
            if "round" in item:
                break
            gazeInfo[i].append(item)
            idx += 1

    return target, gazeInfo


def Rearange(target, gazeInfo, roundNum : int):
    time = [ [] for _ in range(roundNum)]
    headOri = [ [] for _ in range(roundNum)]
    headDir = [ [] for _ in range(roundNum)]
    gazeOri = [ [] for _ in range(roundNum)]
    gazeDir = [ [] for _ in range(roundNum)]
    localGaze = [ [] for _ in range(roundNum)]
    localTarget = [ [] for _ in range(roundNum)]
    
    for i in range(roundNum):
        t = 0.0
        for item in gazeInfo[i]:
            t += item["t"]
            time[i].append(t)
            headOri[i].append( [ item["headOri"]["x"], item["headOri"]["y"], item["headOri"]["z"] ] )
            headDir[i].append( [ item["headDir"]["x"], item["headDir"]["y"], item["headDir"]["z"] ] )
            gazeOri[i].append( [ item["gazeOri"]["x"], item["gazeOri"]["y"], item["gazeOri"]["z"] ] )
            gazeDir[i].append( [ item["gazeDir"]["x"], item["gazeDir"]["y"], item["gazeDir"]["z"] ] )
            localGaze[i].append( [ item["gazeX"], item["gazeY"] ] )
            localTarget[i].append( [ item["targetX"], item["targetY"] ] )
        time[i] = np.array(time[i])
        headOri[i] = np.array(headOri[i])
        headDir[i] = np.array(headDir[i])
        gazeOri[i] = np.array(gazeOri[i])
        gazeDir[i] = np.array(gazeDir[i])
        localGaze[i] = np.array(localGaze[i])
        localTarget[i] = np.array(localTarget[i])

    headPos = [ [] for _ in range(roundNum)]
    gazePos = [ [] for _ in range(roundNum)]
    for i in range(roundNum):
        headPos[i] = ( (target[i][2] - headOri[i][:, 2]) / headDir[i][:, 2] ).reshape(-1, 1) * headDir[i] + headOri[i]
        gazePos[i] = ( (target[i][2] - gazeOri[i][:, 2]) / gazeDir[i][:, 2] ).reshape(-1, 1) * gazeDir[i] + gazeOri[i]

    return {
        "time" : time,
        "headOri" : headOri,
        "headDir" : headDir,
        "gazeOri" : gazeOri,
        "gazeDir" : gazeDir,
        "localGaze" : localGaze,
        "localTarget" : localTarget,
        "headPos" : headPos,
        "gazePos" : gazePos,
    }


class SerializedData:

    ROUNDNUM = 20
    FPS = 60
    
    def __init__(self, filename : str):
        self.username = filename
        self.userid = 1
        
        target, gazeInfo = loadSerializedData(filename, self.ROUNDNUM)
        gazeInfo = Rearange(target, gazeInfo, self.ROUNDNUM)

        self.target = target
        self.gazeInfo = gazeInfo

        self.valid = np.ones(self.ROUNDNUM, dtype=np.int32)
        self.gazeIndex = np.zeros(self.ROUNDNUM, dtype=np.int32)
        self.saccadeDetect()

        self.accuracy = np.zeros((self.ROUNDNUM, 2), dtype=np.float32)
        self.precision = np.zeros((self.ROUNDNUM, 2), dtype=np.float32)
        self.calAccuracy()


    def fetch(self, index : int, ctype : str):
        if ctype == "target":
            return self.target[index]
        return self.gazeInfo[ctype][index]


    def saccadeDetect(self):

        for i in range(self.ROUNDNUM):            
            gazeDir = self.gazeInfo["gazeDir"][i]
            time = self.gazeInfo["time"][i]

            for idx in range(1, gazeDir.shape[0]):
                cosa = np.dot(gazeDir[idx], gazeDir[idx-1]) / (np.linalg.norm(gazeDir[idx]) * np.linalg.norm(gazeDir[idx-1]))
                angle = np.arccos(min(cosa, 1.0))
                angle_psec = angle / (time[idx] - time[idx-1])
                if angle_psec > 3.5:
                    if time[idx] > 1.0:
                        self.valid[i] = 0
                    self.gazeIndex[i] = idx

    def calAccuracy(self):

        for i in range(self.ROUNDNUM):
            if self.valid[i] == 0:
                self.accuracy[i] = np.array([-1.0, -1.0])
                self.precision[i] = np.array([-1.0, -1.0])
                continue
            gazePos = self.gazeInfo["gazePos"][i]
            for idx in range(self.gazeIndex[i], gazePos.shape[0]):
                self.accuracy[i] += gazePos[idx][0:2]
            self.accuracy[i] /= (gazePos.shape[0] - self.gazeIndex[i])
            for idx in range(self.gazeIndex[i], gazePos.shape[0]):
                self.precision[i] += (gazePos[idx][0:2] - self.accuracy[i]) ** 2
            self.precision[i] /= (gazePos.shape[0] - self.gazeIndex[i])
            self.precision[i] = np.sqrt(self.precision[i])
            self.accuracy[i] = np.abs(self.accuracy[i] - self.target[i][0:2])
        
    def showAccuracy(self):
        # print(self.accuracy[self.valid == 1])
        # print(self.precision[self.valid == 1])
        print("average accuracy", np.mean(self.accuracy[self.valid], axis=0))
        print("average precision", np.mean(self.precision[self.valid], axis=0))
        