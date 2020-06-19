#!/bin/bash
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 20 --num-epoch 5
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 20 --num-epoch 100
python Main_Model.py --challenge singlecoil --num-chans 32 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 20 --num-epoch 5
python Main_Model.py --challenge singlecoil --num-chans 32 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 20 --num-epoch 100

