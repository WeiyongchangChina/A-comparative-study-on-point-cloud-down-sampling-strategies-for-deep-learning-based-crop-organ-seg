#!/usr/bin/bash

source /home/david/anaconda3/bin/activate ljs

python test.py --model_path '/media/david/HDD1/ljs_bak/PlantNet-randlaet/models/log_test_10提升/epoch_150.ckpt'
python eval_iou_accuracy.py
