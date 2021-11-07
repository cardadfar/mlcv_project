from sklearn.manifold import TSNE
import json
import numpy as np

data = json.load(open('data.json', 'r'))

X = []
for i in range(len(data["data"])):
    encoding = data["data"][i]["encoding"]
    X.append( encoding )

X_tsne = TSNE(n_components=3).fit_transform(X)

for i in range(len(data["data"])):
    data["data"][i]["encoding"] = X_tsne[i].tolist()

with open('data_tsne.json', 'w') as fp:
    json.dump(data, fp, indent=4)
