import pydicom as dicom
from matplotlib import pyplot, cm
import os
from pydicom.data import get_testdata_file
import numpy as np


def MRI_RawDataExtractor(ScanDataPath,PatientID,PatientDateScan,indices, args):

    if not os.path.exists(os.path.join('Segmentation',ScanDataPath)):
        os.makedirs(os.path.join('Segmentation',ScanDataPath))
    if not os.path.exists(os.path.join('Segmentation',ScanDataPath,'Scan_CherryPick')):
        os.makedirs(os.path.join('Segmentation',ScanDataPath,'Scan_CherryPick'))
    if not os.path.exists(os.path.join(args.image_path)):
        os.makedirs(os.path.join(args.image_path))

    PathDicom = ScanDataPath
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                if "1-" in filename.lower():
                  lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the X axis)
    ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))


    #In case we don;t have SpacingBetweenSlices
    try:
        SpacingBetweenSlices = float(RefDs.SpacingBetweenSlices)
    except:
        SpacingBetweenSlices = float(2.5)

    # Load spacing values (in mm)
    ConstPixelSpacing = (SpacingBetweenSlices, float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])


    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)


    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM),:, :] = ds.pixel_array
        pyplot.imsave(os.path.join('Segmentation', ScanDataPath,"ScanImage-{}-{}-{}.png".format(PatientID,PatientDateScan,lstFilesDCM.index(filenameDCM))), (ArrayDicom[lstFilesDCM.index(filenameDCM),:, :]), cmap='gray')

    for i in indices:
        # Save scans for debugging
        pyplot.imsave(os.path.join('Segmentation', ScanDataPath,'Scan_CherryPick',"{}-{}-{}-ScanImage.png".format(PatientID,PatientDateScan,i)), (ArrayDicom[i,:, :]), cmap='gray')

        # Prepare the Dataset
        pyplot.imsave(os.path.join(args.image_path,"{}-{}-{}-ScanImage.png".format(PatientID, PatientDateScan, i)),(ArrayDicom[i, :, :]), cmap='gray')

    # #Original Scan figure
    # # i = 37
    # pyplot.figure(dpi=300)
    # pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    # pyplot.pcolormesh(y, z, np.flipud(ArrayDicom[i,:, :]))
    # pyplot.show()



    return len(lstFilesDCM)


if __name__ == "__main__":
    Raw = MRI_RawDataExtractor(os.path.join('DataBase','ISPY1','ISPY1_1009','03-16-1985-485859-MR BREASTUNI UE-34504','3.000000-Dynamic-3dfgre-93714'))
    print('Finish')