import pydicom as dicom
from matplotlib import pyplot, cm
import os
from pydicom.data import get_testdata_file
import numpy as np
import cv2
import logging

def MRI_SegentationDataExtractor(SegmentationDataPath, SegmentationMaskDataPath, PatientID, PatientDateScan,args):

    if not os.path.exists(os.path.join('Segmentation',SegmentationDataPath,'WOMorph')):
        os.makedirs(os.path.join('Segmentation',SegmentationDataPath,'WOMorph'))
    if not os.path.exists(os.path.join('Segmentation', SegmentationDataPath, 'WMorph')):
        os.makedirs(os.path.join('Segmentation', SegmentationDataPath,'WMorph'))
    if not os.path.exists(os.path.join('Segmentation', SegmentationDataPath,'AdaptiveThresh')):
        os.makedirs(os.path.join('Segmentation', SegmentationDataPath,'AdaptiveThresh'))
    if not os.path.exists(os.path.join('Segmentation', SegmentationDataPath,'AT-CP')):
        os.makedirs(os.path.join('Segmentation', SegmentationDataPath,'AT-CP'))
    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)


    PathDicom = SegmentationDataPath
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower(): # check whether the file's DICOM
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
    MorphArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    ds = dicom.read_file(lstFilesDCM[0])

    PictureRowDirection = np.asarray(ds[0x0020, 0x0037]._value[0:3], dtype=int)
    PictureColumDirection = np.asarray(ds[0x0020, 0x0037]._value[3:6], dtype=int)
    ScaningDirection = np.cross(PictureRowDirection, PictureColumDirection)
    # print('Scanning Direction', ScaningDirection)


    ##########################################################
    ##########################################################

    # The array is sized based on 'ConstPixelDims'
    Seg_ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    #Segmentation Layer
    Seg_lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(SegmentationMaskDataPath):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                Seg_lstFilesDCM.append(os.path.join(dirName,filename))

    ds_seg = dicom.dcmread(Seg_lstFilesDCM[0])

    #Calc the Segmentation VOI center with the original scan origin
    Delta_VOI_Center = -np.array(ds[0x0020,0x0032]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1042]._value)[0:3]
    Seg_Loc_Pixel_Slice = np.round(Delta_VOI_Center / np.array(ConstPixelSpacing))



    Delta = np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1043]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1044]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1045]._value)
    Seg_Half_Box = np.round(Delta[0:3] / np.array(ConstPixelSpacing))


    Seg_x = np.arange(Seg_Loc_Pixel_Slice[0]-Seg_Half_Box[0],Seg_Loc_Pixel_Slice[0]+Seg_Half_Box[0],1, dtype=int) * ScaningDirection[0]
    Seg_y = np.arange(Seg_Loc_Pixel_Slice[1]-Seg_Half_Box[1],Seg_Loc_Pixel_Slice[1]+Seg_Half_Box[1],1,dtype=int) * PictureRowDirection[1]
    Seg_z = np.arange(Seg_Loc_Pixel_Slice[2]-Seg_Half_Box[2],Seg_Loc_Pixel_Slice[2]+Seg_Half_Box[2],1,dtype=int) * PictureColumDirection[2]

    # print('Segmented slices:', Seg_x)

    Seg_ArrayDicom[min(Seg_x):max(Seg_x),min(Seg_y):max(Seg_y),min(Seg_z):max(Seg_z)] = 1
    ######
    PicturesHistograms = np.zeros(len(lstFilesDCM), dtype=int)
    PreprocessingMedicalImage = []

    # loop through all the DICOM files to find scan dynamic range
    DynamicRangeMin = 0
    DynamicRangeMax = 0
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM),:, :] = ds.pixel_array

        if np.min(ArrayDicom[lstFilesDCM.index(filenameDCM),:, :]) < DynamicRangeMin:
            DynamicRangeMin = np.min(ArrayDicom[lstFilesDCM.index(filenameDCM),:, :])
        if np.max(ArrayDicom[lstFilesDCM.index(filenameDCM),:, :]) > DynamicRangeMax:
            DynamicRangeMax = np.max(ArrayDicom[lstFilesDCM.index(filenameDCM),:, :])

    ImageThreshold = (DynamicRangeMax - DynamicRangeMin)*0.1

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM),:, :] = ds.pixel_array

        kernel = np.ones((3, 3), np.uint8)

        #Doing morphological operation, first I'm doing thresholding for the image then I used open (dilute and then erode) to remove noise and close the black pixels inside the lesions
        PreprocessingMedicalImage = cv2.morphologyEx(cv2.threshold(ArrayDicom[lstFilesDCM.index(filenameDCM), :, :].astype('uint8'), ImageThreshold, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN,kernel)
        # PreprocessingMedicalImage.append(cv2.morphologyEx(cv2.threshold(ArrayDicom[lstFilesDCM.index(filenameDCM), :, :].astype('uint8'), 30, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN, kernel))

        #Histogram of pixel summary normalized by 255
        PicturesHistograms[lstFilesDCM.index(filenameDCM)] = np.sum(PreprocessingMedicalImage) / 255



        #Doing OPEN and CLOSE sequentially without using threshold before
        MorphArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = cv2.morphologyEx(ArrayDicom[lstFilesDCM.index(filenameDCM), :, :], cv2.MORPH_OPEN, kernel)
        MorphArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = cv2.morphologyEx(MorphArrayDicom[lstFilesDCM.index(filenameDCM), :, :], cv2.MORPH_CLOSE, kernel)

        #Saving the three configuration
        pyplot.imsave(os.path.join('Segmentation',SegmentationDataPath,'WOMorph',"SegImage-{}-{}-{}.png".format(PatientID,PatientDateScan,lstFilesDCM.index(filenameDCM))), (ArrayDicom[lstFilesDCM.index(filenameDCM), :, :]), cmap='gray')
        pyplot.imsave(os.path.join('Segmentation',SegmentationDataPath,'WMorph',"SegImage-{}-{}-{}.png".format(PatientID,PatientDateScan,lstFilesDCM.index(filenameDCM))), (MorphArrayDicom[lstFilesDCM.index(filenameDCM), :, :]), cmap='gray')
        pyplot.imsave(os.path.join('Segmentation',SegmentationDataPath,'AdaptiveThresh',"SegImage-{}-{}-{}.png".format(PatientID,PatientDateScan,lstFilesDCM.index(filenameDCM))),PreprocessingMedicalImage,cmap='gray')


    Precentage = 0.8  # 80% deviation from the maximum value
    MinPixels = 100
    # Saving the indices for the most segnificant pictures
    indices = []
    for i, ScanIndex in enumerate(PicturesHistograms):
        if ScanIndex > (1 - Precentage) * PicturesHistograms.max():
            if ScanIndex > MinPixels:
                indices.append(i)
                #Save for scans debugging
                pyplot.imsave(os.path.join('Segmentation', SegmentationDataPath,'AT-CP',"{}-{}-{}-SegImage.png".format(PatientID, PatientDateScan,i)),
                          cv2.morphologyEx(cv2.threshold(ArrayDicom[i, :, :].astype('uint8'), ImageThreshold, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN,kernel), cmap='gray')

                #Prepare the Dataset
                pyplot.imsave(os.path.join(args.mask_path,"{}-{}-{}-SegImage.png".format(PatientID, PatientDateScan, i)),
                              cv2.morphologyEx(cv2.threshold(ArrayDicom[i, :, :].astype('uint8'), ImageThreshold, 255,cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN, kernel),cmap='gray')
    #
    print('Most significant scans are:', indices)
    # pyplot.figure(dpi=300)
    # pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    # pyplot.imshow(cv2.morphologyEx(cv2.threshold(ArrayDicom[27, :, :].astype('uint8'), 30, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN,kernel))
    # pyplot.show()

    if not indices:
        logging.info('Not extracted scans for %s', PatientID)

    # #Segmentation Scan figure
    # # i = 27
    #
    # # Pre-process Morpholigical tool
    # kernel = np.ones((4, 4), np.uint8)
    # ArrayDicom[i,:, :] = cv2.morphologyEx(ArrayDicom[i,:, :], cv2.MORPH_OPEN, kernel)
    # ArrayDicom[i,:, :] = cv2.morphologyEx(ArrayDicom[i,:, :], cv2.MORPH_CLOSE, kernel)
    #
    # pyplot.figure(dpi=300)
    # pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    # pyplot.pcolormesh(y, z, np.flipud(ArrayDicom[i,:, :]))
    # pyplot.show()
    #
    # # #Segmentation with Mask Scan figure
    # # pyplot.figure(dpi=300)
    # # pyplot.axes().set_aspect('equal', 'datalim')
    # # pyplot.set_cmap(pyplot.gray())
    # # pyplot.pcolormesh(y, z, np.flipud(ArrayDicom[i,:, :])*np.fliplr(Seg_ArrayDicom[i,:,:]))
    # # pyplot.show()
    #
    # os.mkdir(os.path.join('Segmentation', SegmentationDataPath))
    # pyplot.imsave("Test\SegmentationImage-{}.png".format(i), (ArrayDicom[i, :, :]),cmap='gray')
    return indices, len(lstFilesDCM)

if __name__ == "__main__":
    SegmentationDataPath = os.path.join('DataBase','ISPY1', 'ISPY1_1009', '03-16-1985-485859-MR BREASTUNI UE-34504', '31000.000000-Dynamic-3dfgre SER-42809')
    SegmentationMaskDataPath = os.path.join('DataBase','ISPY1', 'ISPY1_1009', '03-16-1985-485859-MR BREASTUNI UE-34504', '32001.000000-Breast Tissue Segmentation-29336')
    Raw = MRI_SegentationDataExtractor(SegmentationDataPath, SegmentationMaskDataPath)
    print('Finish')