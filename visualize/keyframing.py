from sklearn.decomposition import PCA
import json
import numpy as np

data = json.load(open('data.json', 'r'))

X = []

idx1 = 10
idx2 = 11
keyframe1 = []
keyframe2 = []
for i in range(len(data["data"])):
    encoding = data["data"][i]["encoding"]
    if i == idx1:
        keyframe1.append( encoding )
    if i == idx2:
        keyframe2.append( encoding )

nSteps = 10
Xt = []
for i in range(len(keyframe1)):
    for ti in range(nSteps+1):
        t = ti / nSteps
        interp = (1 - t) * np.asarray(keyframe1[i]) + (t) * np.asarray(keyframe2[i])
        Xt.append( interp )

data["interp"] = []
for i in range(len(keyframe1)):
    data["interp"].append( [] )
    for ti in range(nSteps+1):
        obj = {}
        obj["encoding"] = Xt[ti].tolist()
        data["interp"][i].append( obj )

with open('data_keyframe.json', 'w') as fp:
    json.dump(data, fp, indent=4)
