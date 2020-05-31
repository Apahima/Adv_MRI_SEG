import pydicom as dicom
import matplotlib.pylab as plt
import os


# specify your image path
image_path = [r'RawData\ISPY1_1002\11-02-1984-868859-MR BREASTUNI UE-60097\2.000000-T2-FSE-Sagittal-51323\1-09.dcm',
 r'RawData\ISPY1_1002\11-02-1984-868859-MR BREASTUNI UE-60097\2.000000-T2-FSE-Sagittal-51323\1-10.dcm'
,r'RawData\ISPY1_1002\11-02-1984-868859-MR BREASTUNI UE-60097\32101.000000-VOI Breast Tissue Segmentation-96213\1-1.dcm']

ds = dicom.dcmread(image_path[3])

#Image Orientation
print(ds[0x0020,0x0037])
# Image Position
print(ds[0x0020,0x0032])
#Spacing between slices
print(ds[0x0018,0x0088])
#Pixel spcaing
print(ds[0x0028,0x0030])

#Segmentation

#VOI_Pixel_Start
print('VOI_Pixel_Start', ds[0x0117,0x10a1])
#VOI_Pixel_End
print('VOI_Pixel_End', ds[0x0117, 0x10a2])

# a = ds[0x17,1xA1][0]

plt.imshow(ds.pixel_array, cmap='gray', vmin=0, vmax=255)
plt.show()
print('End')