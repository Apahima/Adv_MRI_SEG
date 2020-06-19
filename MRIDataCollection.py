import os
import logging
from MRI_BreastRawDataScanExtractor import MRI_RawDataExtractor as MRI_Ex
from MRI_BreastSegmentationDataScanExtractor import MRI_SegentationDataExtractor as MRI_Seg
from Common.RawDataPreProcessing_Medical_Parse import RawDataPreProcessing_Args as RawDataParsing

logging.basicConfig(level=logging.DEBUG,filemode='w', filename=os.getcwd()+'/Debug.log')

# OriginalScan = os.path.join('DataBase','ISPY1','ISPY1_1009','03-16-1985-485859-MR BREASTUNI UE-34504','3.000000-Dynamic-3dfgre-93714')
# SegmentationDataPath, SegmentationMaskDataPath = (os.path.join('DataBase','ISPY1', 'ISPY1_1009', '03-16-1985-485859-MR BREASTUNI UE-34504', '31001.000000-Dynamic-3dfgre PE1-39677')
#     ,os.path.join('DataBase','ISPY1', 'ISPY1_1009', '03-16-1985-485859-MR BREASTUNI UE-34504', '32001.000000-Breast Tissue Segmentation-29336'))

args = RawDataParsing().parse_args()  #Pulling the argument for using during the code
PathDataset = args.data_path

# a = [x[0] for x in os.walk(PathDataset)]

for PatientID in os.listdir(PathDataset):
    print(PatientID)
    try:
        for PatientDateScan in os.listdir(os.path.join(PathDataset,PatientID)):
            PatientFilesPath = [x[0] for x in os.walk(os.path.join(PathDataset,PatientID,PatientDateScan))]

            RawDataScanIndeces = []
            PESegDataScanIndeces = []
            SERSegDataScanIndeces = []
            BreastTissueSegDataScanIndeces = []

            for i, File in enumerate(PatientFilesPath):
                # if 'Dynamic-3dfgre' in File:
                #     if 'PE1' not in File:
                #         if 'SER' not in File:
                #             RawDataScanIndeces = i
                if 'PE' in File:
                    PESegDataScanIndeces = i
                elif 'SER' in File:
                    SERSegDataScanIndeces = i
                elif 'Breast Tissue Segmentation' in File:
                    if 'VOI' not in File:
                        BreastTissueSegDataScanIndeces = i
                else:
                    RawDataScanIndeces = i


            print('Folder With data indices- (RawData,PE1,SER,BreastTissueSegmentation): {}-{}-{}-{}'.format(RawDataScanIndeces,PESegDataScanIndeces,SERSegDataScanIndeces,BreastTissueSegDataScanIndeces))



            #         indices.append(i)
            # RawDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre' in s]
            # PESegDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre' & 'PE1'  in s]
            # SERSegDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre' & 'SER' in s]
            # BreastTissueSegDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Breast Tissue Segmentation' in s]

            print(PatientFilesPath[0])
            indices, SegmentationScansNumber = MRI_Seg(PatientFilesPath[PESegDataScanIndeces],PatientFilesPath[BreastTissueSegDataScanIndeces],PatientID,PatientDateScan, args)    #PE Segmentation
            #_ = MRI_Seg(PatientFilesPath[SERSegDataScanIndeces], PatientFilesPath[BreastTissueSegDataScanIndeces],PatientID,PatientDateScan)  #Optional SER Segmentation
            BreastScansNumber = MRI_Ex(PatientFilesPath[RawDataScanIndeces], PatientID, PatientDateScan,indices, args)

            assert SegmentationScansNumber == BreastScansNumber #Checking whether the Scans images are equal to the segmentation slice

    except AssertionError as error:
        logging.debug('{}-Scan and Segmentation are Unequal'.format(PatientID))

    except Exception as ex:
        # print('Unable to Pre-Processing Patient:', PatientID)
        # print(ex)
        logging.info('Unable to Pre-Processing Patient: %s', PatientID)
        logging.debug(ex)
        logging.debug('Scan and Segmentation are Unequal')


print('Finish')

#
# for dirName, subdirList, fileList in os.walk(PathDataset):
#     print(dirName)
#     print(subdirList)
#     print(fileList)
    # for SubDirFolder in subdirList: #For Each Patient
    #     print(SubDirFolder)
    #     for SubdirName, SubsubdirList, SubfileList in os.walk(os.path.join(dirName,SubDirFolder)):
    #         print(SubdirName)
    #         print(SubsubdirList)
    #         print(SubfileList)
    #         for SubDirFolderDate in SubsubdirList:  # For Each Patient and Date
    #             print(SubDirFolderDate)
    #             PatientFilesPath = [x[0] for x in os.walk(os.path.join(dirName,SubDirFolder,SubDirFolderDate))]
    #             RawDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre-' in s]
    #             PESegDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre PE1' in s]
    #             SERSegDataScanIndeces = [i for i, s in enumerate(PatientFilesPath) if 'Dynamic-3dfgre SER' in s]




#
# for i in range(0,60,1):
#     # MRI_Ex(OriginalScan, i)
#     MRI_Seg(SegmentationDataPath, SegmentationMaskDataPath, i)