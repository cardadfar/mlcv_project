import cv2
import numpy as np
import os
import json

def img2vid(img_dir, frame_size=(28, 28), output_path='video.avi'):
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, frame_size)
    filenames = os.listdir(img_dir)

    filenames = sorted(filenames, key=lambda x: int(x[:-4]))

    for filename in filenames:
        img = cv2.imread(img_dir + filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        video.write(img)

    video.release()

def save2json(img_paths, classIDs, encodings, tweens):
    '''
    img_paths: (N) paths to images
    classIDs: (N) IDs of each input class
    encodings: (N x embedding) latent-space representation
    tweens: (A x B x embedding) multiple interpolation values for visualizing
    '''

    N = len(img_paths)

    data = []
    nClasses = 0
    for i in range(N):
        obj = {}
        obj["id"] = classIDs[i]
        obj["img_path"] = img_paths[i]
        obj["encoding"] = encodings[i]
        data.append(obj)

        nClasses = max(nClasses, classIDs[i])
        dims = len(encodings[i])

    interp = []
    for i in range(len(tweens)):
        interp.append([])
        for j in range(len(tweens[i])):
            obj = {}
            obj["encoding"] = tweens[i][j]
            interp[i].append(obj)
    
    results = {}
    results["dims"] = dims
    results["nclasses"] = nClasses
    results["data"] = data
    results["interp"] = interp

    with open('visualize/data.json', 'w') as fp:
        json.dump(results, fp)


img2vid('results/test/')
