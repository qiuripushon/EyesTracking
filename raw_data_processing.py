import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from SerializedData import *

OFFSET = 10

if __name__ == "__main__":
    filenames = glob("raw_data/*.json")

    local_data = pd.DataFrame()
    global_data = pd.DataFrame()

    global_data_zzp = np.array([])
    local_data_zzp = pd.DataFrame()

    for filename in tqdm(filenames):
        try:
            sdata = SerializedData(filename)
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

        for i in range(20):
            if sdata.valid[i] == 0:
                print('nnnno')
                continue

            headPos = sdata.fetch(i, "headPos")
            print('yes')

            temp1 = np.zeros((1, 300))
            temp2 = np.zeros((1, 300))
            gazePos = sdata.fetch(i, "gazePos")
            time = sdata.fetch(i, "time")
            target = sdata.fetch(i, "target")

            start = sdata.gazeIndex[i]
            end = headPos.shape[0]
            num = end - start + 1

            headV = np.zeros((end, 3))
            gazeV = np.zeros((end, 3))
            temp1[0, 0] = target[0]
            temp2[0, 0] = target[1]

            for iii in range(start, min(end, start + 299)):
                temp1[0, iii - start + 1] = gazePos[iii, 0]
                temp2[0, iii - start + 1] = gazePos[iii, 1]

            for idx in range(start + 1, end):
                headV[idx] = (headPos[idx] - headPos[idx - 1]) / (time[idx] - time[idx - 1])
                gazeV[idx] = (gazePos[idx] - gazePos[idx - 1]) / (time[idx] - time[idx - 1])

            gdata = {
                "gazeX": gazePos[start:end, 0],
                "gazeY": gazePos[start:end, 1]
            }

            if sdata.valid[i] != 0:
                global_data_zzp = np.vstack([global_data_zzp, temp1]) if global_data_zzp.size else temp1
                global_data_zzp = np.vstack([global_data_zzp, temp2]) if global_data_zzp.size else temp2

            localGaze = sdata.fetch(i, "localGaze")
            localTarget = sdata.fetch(i, "localTarget")
            localGazeV = np.zeros((end, 2))
            for idx in range(start + 1, end):
                localGazeV[idx] = (localGaze[idx] - localGaze[idx - 1]) / (time[idx] - time[idx - 1])

            ldata = {
                "diffX": localTarget[start + OFFSET:end, 0] - localGaze[start + OFFSET:end, 0],
                "diffY": localTarget[start + OFFSET:end, 1] - localGaze[start + OFFSET:end, 1],
            }
            for j in range(10):
                ldata["localGazeX_" + str(j)] = localGaze[start + OFFSET - j:end - j, 0]
                ldata["localGazeY_" + str(j)] = localGaze[start + OFFSET - j:end - j, 1]
                ldata["localGazeVX_" + str(j)] = localGazeV[start + OFFSET - j:end - j, 0]
                ldata["localGazeVY_" + str(j)] = localGazeV[start + OFFSET - j:end - j, 1]

            local_data = pd.concat([local_data, pd.DataFrame(ldata)])

    local_data_zzp.to_csv("local_data_zzp1.csv")
    np.savetxt("savefile.csv", global_data_zzp, delimiter=",")
