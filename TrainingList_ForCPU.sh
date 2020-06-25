#!/bin/bash
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --lr 0.0001 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --lr 0.05 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --lr 0.0001 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 5 --batch-size 10 --lr 0.05 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --lr 0.0001 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 64 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --lr 0.05 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --lr 0.0001 --num-epoch 5 --exp-dir AlmFinal
python Main_Model.py --challenge singlecoil --num-chans 32 --device cpu --drop-prob 0.5 --data-path Temp/ISPY1  --num-pools 3 --batch-size 10 --lr 0.05 --num-epoch 5 --exp-dir AlmFinal