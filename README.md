# Adv_MRI_SEG

1. Download the scans from: https://www.cancerimagingarchive.net/nbia-search/?CollectionCriteria=ISPY1. The scans are order chronoligicaly, needs to download only the erlier scan for each patient (Total 200 patients).Need to download specific files from each series, the files are:
* Full scan
* Scan with PE
* Scan with SER
* Breast Tissue Segmentation 
2. Place the file at: {Working Directory}\DataBase
3. For Pre-Processing run MRIDataCollection.py
    python MRIDataCollection.py --data-path 
4. For running training phase run Main_Model.py
    python Main_Model.py --
    
The results store as TensorBoardX files, to be able to see the results, luanch TensorBoard from checkpoint folder
    tensorboard --logdir checkpoint --host {HostIP} --port {Port}
Open your browser and type IP:Port, tensor Board should open and display the results
