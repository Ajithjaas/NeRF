import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pry

from DataLoader import *
from Train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def main():
    # testing data loading
    data = DataLoader("lego/")
    trainImages, trainTransMat, trainRot = data.load_data('train')

    focalLength = np.array([138.8888789])
    focalLength = torch.from_numpy(focalLength).to(device)

    height, width = trainImages.shape[1:3]

    camera_view_bounds = {
        "near": 2,
        "far": 6
    }

    trainTransMat = torch.from_numpy(trainTransMat).to(device)
    trainImages = torch.from_numpy(trainImages[:, ..., :3]).to(device)

    print("Starting training")
    model = Train(height, width, trainImages, trainTransMat, focalLength, camera_view_bounds)
    # print(type(model))
    model.train(6,4,4)


if __name__ == "__main__":
    main()
