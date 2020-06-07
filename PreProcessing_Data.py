import pydicom as dicom
from matplotlib import pyplot, cm
import os
from pydicom.data import get_testdata_file
import numpy as np


os.path.join('RawData','ISPY1_1002','11-02-1984-868859-MR BREASTUNI UE-60097','31000.000000-Dynamic-3dfgre SER-66206')

PathDicom = os.path.join('RawData','ISPY1_1002','11-02-1984-868859-MR BREASTUNI UE-60097','31000.000000-Dynamic-3dfgre SER-66206')
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the X axis)
ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.SpacingBetweenSlices), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])


# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)


#####
# The array is sized based on 'ConstPixelDims'
Seg_ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
#Segmentation Layer
image_path = [r'RawData\ISPY1_1002\11-02-1984-868859-MR BREASTUNI UE-60097\32001.000000-Breast Tissue Segmentation-93996\1-1.dcm']
ds_seg = dicom.dcmread(image_path[0])

#Calc the Segmentation VOI center
ds = dicom.read_file(lstFilesDCM[0])

Delta_VOI_Center = -np.array(ds[0x0020,0x0032]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1042]._value)[0:3]
Seg_Loc_Pixel_Slice = np.round(np.abs(Delta_VOI_Center / np.array(ConstPixelSpacing)))



Delta = np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1043]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1044]._value)+np.array(ds_seg[0x0117,0x1020]._value[0][0x0117,0x1045]._value)
Seg_Half_Box = np.round(Delta[0:3] / np.array(ConstPixelSpacing))

Seg_x = np.arange(Seg_Loc_Pixel_Slice[0]-Seg_Half_Box[0],Seg_Loc_Pixel_Slice[0]+Seg_Half_Box[0],1, dtype=int)
Seg_y = np.arange(Seg_Loc_Pixel_Slice[1]-Seg_Half_Box[1],Seg_Loc_Pixel_Slice[1]+Seg_Half_Box[1],1,dtype=int)
Seg_z = np.arange(Seg_Loc_Pixel_Slice[2]-Seg_Half_Box[2],Seg_Loc_Pixel_Slice[2]+Seg_Half_Box[2],1,dtype=int)

Seg_ArrayDicom[min(Seg_x):max(Seg_x),min(Seg_y):max(Seg_y),min(Seg_z):max(Seg_z)] = 1
######



# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[lstFilesDCM.index(filenameDCM),:, :] = ds.pixel_array


#Original Scan figure
i = 37

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(y, z, np.flipud(ArrayDicom[i,:, :]))
pyplot.show()


#Original Scan figure
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(y, z, np.flipud(ArrayDicom[i,:, :]*np.fliplr(Seg_ArrayDicom[i,:,:])))
pyplot.show()

# specify your image path
image_path = [r'RawData\ISPY1_1002\11-02-1984-868859-MR BREASTUNI UE-60097\32101.000000-VOI Breast Tissue Segmentation-96213\1-1.dcm']

ds_seg = dicom.dcmread(image_path[0])

# #Image Orientation
# print(ds[0x0020,0x0037])
# # Image Position
# print(ds[0x0020,0x0032])
# #Spacing between slices
# print(ds[0x0018,0x0088])
# #Pixel spcaing
# print(ds[0x0028,0x0030])

#Segmentation

#VOI_Pixel_Start
print('VOI_Pixel_Start', ds_seg[0x0117,0x10a1])
#VOI_Pixel_End
print('VOI_Pixel_End', ds_seg[0x0117, 0x10a2])

# a = ds[0x17,1xA1][0]

# plt.imshow(ds.pixel_array, cmap='gray', vmin=0, vmax=255)
# plt.show()
# print('End')Seg_ArrayDicom