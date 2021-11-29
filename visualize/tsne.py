from sklearn.manifold import TSNE
import json
import numpy as np

data = json.load(open('data.json', 'r'))

USEINTERP = False
if "interp" in data:
    USEINTERP = True

X = []
for i in range(len(data["data"])):
    encoding = data["data"][i]["encoding"]
    X.append( encoding )

n = len(X)
if USEINTERP:
    for i in range(len(data["interp"])):
        for j in range(len(data["interp"][i])):
            encoding = data["interp"][i][j]["encoding"]
            X.append( encoding )
            
X_tsne = TSNE(n_components=3).fit_transform(X)

for i in range(len(data["data"])):
    data["data"][i]["encoding"] = X_tsne[i].tolist()

if USEINTERP:
    k = 0
    for i in range(len(data["interp"])):
        for j in range(len(data["interp"][i])):
            data["interp"][i][j]["encoding"] = X_tsne[k + n].tolist()
            k += 1

with open('data_tsne.json', 'w') as fp:
    json.dump(data, fp, indent=4)
