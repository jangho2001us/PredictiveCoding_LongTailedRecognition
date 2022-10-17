#!/bin/bash

python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.005 --batch-size 128 --loss_type BMS --seed 0 --exp_str 0 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.005 --batch-size 128 --loss_type BMS --seed 1 --exp_str 1 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.005 --batch-size 128 --loss_type BMS --seed 2 --exp_str 2 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.005 --batch-size 128 --loss_type BMS --seed 3 --exp_str 3 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.005 --batch-size 128 --loss_type BMS --seed 4 --exp_str 4 --lr 0.1

python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.01 --batch-size 128 --loss_type BMS --seed 0 --exp_str 0 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.01 --batch-size 128 --loss_type BMS --seed 1 --exp_str 1 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.01 --batch-size 128 --loss_type BMS --seed 2 --exp_str 2 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.01 --batch-size 128 --loss_type BMS --seed 3 --exp_str 3 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.01 --batch-size 128 --loss_type BMS --seed 4 --exp_str 4 --lr 0.1

python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.02 --batch-size 128 --loss_type BMS --seed 0 --exp_str 0 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.02 --batch-size 128 --loss_type BMS --seed 1 --exp_str 1 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.02 --batch-size 128 --loss_type BMS --seed 2 --exp_str 2 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.02 --batch-size 128 --loss_type BMS --seed 3 --exp_str 3 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.02 --batch-size 128 --loss_type BMS --seed 4 --exp_str 4 --lr 0.1

python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.1 --batch-size 128 --loss_type BMS --seed 0 --exp_str 0 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.1 --batch-size 128 --loss_type BMS --seed 1 --exp_str 1 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.1 --batch-size 128 --loss_type BMS --seed 2 --exp_str 2 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.1 --batch-size 128 --loss_type BMS --seed 3 --exp_str 3 --lr 0.1
python mnist_train_bp.py --gpu 0 --imb_type exp --imb_factor 0.1 --batch-size 128 --loss_type BMS --seed 4 --exp_str 4 --lr 0.1
