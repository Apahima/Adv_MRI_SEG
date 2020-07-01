#!/bin/bash
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss WBCE_DiceLoss

python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss Tversky --tversky-alpha 0.1 --tversky-beta 0.9
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss Tversky --tversky-alpha 0.2 --tversky-beta 0.8
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss Tversky --tversky-alpha 0.3 --tversky-beta 0.7
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss Tversky --tversky-alpha 0.4 --tversky-beta 0.6
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 10 --exp-dir 28062020 --num-epoch 50 --loss Tversky --tversky-alpha 0.5 --tversky-beta 0.5


