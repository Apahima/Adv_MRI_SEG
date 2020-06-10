import cv2
import glob
import numpy as np
from pathlib import Path as Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, datasets, models

class MedicalDataloaderConstruct(Dataset):

    def __init__(self, input_images, target_masks):
        self.input_images = input_images
        self.target_masks = target_masks

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        X_data = self.input_images[idx]
        Y_data = self.target_masks[idx]

        return [X_data, Y_data]

def MedicalDataLoading(path):
    X_data = []
    files = glob.glob(str(Path.joinpath(path , 'Image')) + '/' + '*.png')
    for myFile in files:
        print(myFile)
        image = np.expand_dims(cv2.imread(myFile, cv2.IMREAD_GRAYSCALE), axis=0).astype('float32') #Grayscale loading but dim #1 save to be compatible with Unet loading
        X_data.append(image)
    print('X_data shape:', np.array(X_data).shape)

    Y_data = []
    files = glob.glob(str(Path.joinpath(path , 'Mask')) + '/' + '*.png')
    for myFile in files:
        print(myFile)
        image = np.expand_dims(cv2.imread(myFile, cv2.IMREAD_GRAYSCALE), axis=0).astype('float32') #Grayscale loading but dim #1 save to be compatible with Unet loading
        Y_data.append(image)

    print('Y_data shape:', np.array(Y_data).shape)

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)

    Train = MedicalDataloaderConstruct(X_train, y_train)
    Val = MedicalDataloaderConstruct(X_test, y_test)


    return Train, Val

if __name__ == '__main__':
    train, val = MedicalDataLoading(Path(r'..\Data\ISPY1'))

    print('Loading Sucess')



