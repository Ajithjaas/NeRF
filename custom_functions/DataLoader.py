import numpy as np
import json
import cv2
import os

class DataLoader():
    def __init__(self, path):
        self.path = path
    
    def load_data(self,split):
        with open(os.path.join(self.path, 'transforms_{}.json'.format(split)), 'r') as fp:
            meta = json.load(fp)

        imgs = []
        transform_matrix = []
        rotation = []
        count = 0
        for frame in meta['frames']:
            fname = frame['file_path'] + '.png'
            img_path = self.path+fname.lstrip(".")
            imgs.append(cv2.imread(img_path))
            transform_matrix.append(np.array(frame['transform_matrix']))
            rotation.append(np.array(frame['rotation']))
            count += 1

            if count == 50:
                break
            
        imgs             = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        transform_matrix = np.array(transform_matrix).astype(np.float32)
        rotation         = np.array(rotation).astype(np.float32)

        return imgs, transform_matrix, rotation