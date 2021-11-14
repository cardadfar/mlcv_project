from sklearn.decomposition import PCA
import json
import numpy as np

data = json.load(open('data_keyframe.json', 'r'))

USEINTERP = False
if "interp" in data:
    USEINTERP = True

X = []
Xt = []
for i in range(len(data["data"])):
    encoding = data["data"][i]["encoding"]
    X.append( encoding )

if USEINTERP:
    for i in range(len(data["interp"])):
        for j in range(len(data["interp"][i])):
            encoding = data["interp"][i][j]["encoding"]
            Xt.append( encoding )

pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)
if USEINTERP:
    Xt_pca = pca.transform(Xt)

for i in range(len(data["data"])):
    data["data"][i]["encoding"] = X_pca[i].tolist()

if USEINTERP:
    k = 0
    for i in range(len(data["interp"])):
        for j in range(len(data["interp"][i])):
            data["interp"][i][j]["encoding"] = Xt_pca[k].tolist()
            k += 1

with open('data_pca.json', 'w') as fp:
    json.dump(data, fp, indent=4)
