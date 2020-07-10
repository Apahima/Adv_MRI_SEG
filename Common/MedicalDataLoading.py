import cv2
import glob
import numpy as np
from pathlib import Path as Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, datasets, models
import logging

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
        # print(myFile)
        image = np.expand_dims(cv2.imread(myFile, cv2.IMREAD_GRAYSCALE), axis=0).astype('float32') #Grayscale loading but dim #1 save to be compatible with Unet loading
        if image.shape == (1,256,256):
            X_data.append(image)
        elif image.shape == (1,512,512):
            dst = cv2.pyrDown(image[0,:,:],dstsize=(256,256))
            dst = np.expand_dims(dst,0)
            X_data.append(dst)
            logging.info('File {} Convert to (256,256)'.format(myFile))
        else:
            logging.debug('File {} not added to Dataloader'.format(myFile))
            logging.debug('Image size might be different that (256,256). The size actuarl image size %s', image.shape)
    print('X_data shape:', np.array(X_data).shape)

    Y_data = []


    files = glob.glob(str(Path.joinpath(path , 'Mask')) + '/' + '*.png')
    for myFile in files:
        # print(myFile)
        image = np.expand_dims(cv2.imread(myFile, cv2.IMREAD_GRAYSCALE), axis=0).astype('float32') #Grayscale loading but dim #1 save to be compatible with Unet loading
        image = 1/(1 + np.exp(-image)) # Since the dynamic range for mask is 0-255 I wouldlike to rescale it to 0-1 to fit model output by sigmoid and dynamic range
        image -= image.min()
        image /= image.max()

        if image.shape == (1,256,256):
            Y_data.append(image)
        elif image.shape == (1,512,512):
            dst = cv2.pyrDown(image[0,:,:],dstsize=(256,256))
            dst = np.expand_dims(dst,0)
            Y_data.append(dst)
            logging.info('File {} Convert to (256,256)'.format(myFile))
        else:
            logging.debug('File {} not added to Dataloader'.format(myFile))
            logging.debug('Image size might be different that (256,256). The size actuarl image size %s', image.shape)

    print('Y_data shape:', np.array(Y_data).shape)

    # X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=1) #Split to 80% Train, And 20% Validation and Test
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=1)  #Split 10% train 10 Validation

    Train = MedicalDataloaderConstruct(X_train, y_train)
    Val = MedicalDataloaderConstruct(X_val, y_val)
    Test = MedicalDataloaderConstruct(X_test, y_test)


    return Train, Val, Test

if __name__ == '__main__':
    train, val = MedicalDataLoading(Path(r'..\Data\ISPY1'))

    print('Loading Sucess')



