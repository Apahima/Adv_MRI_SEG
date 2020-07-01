#!/bin/bash
python Main_Model.py --challenge singlecoil --num-chans 64 --drop-prob 0.5 --data-path Data/ISPY1  --num-pools 5 --batch-size 20 --exp-dir 28062020 --num-epoch 50 --loss Tversky --alpha-tversky 0.1 --beta-tversky 0.9

