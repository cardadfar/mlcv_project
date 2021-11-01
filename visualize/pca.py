from sklearn.decomposition import PCA
import json
import numpy as np

data = json.load(open('data.json', 'r'))

X = []
for i in range(len(data["data"])):
    encoding = data["data"][i]["encoding"]
    X.append( encoding )
    X.append( encoding )

pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)

for i in range(len(data["data"])):
    data["data"][i]["encoding"] = X_pca[i].tolist()

with open('data.json', 'w') as fp:
    json.dump(data, fp, indent=4)
