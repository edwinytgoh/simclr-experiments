python /home/goh/Documents/D3M/simclr-experiments/simclr/finetune.py \
 /home/goh/Documents/D3M/Mars_Classification/msl-labeled-data-set-v2.1 \
 --file_list paper_2900_train.txt \
 --dataset Mars_v2 \
 --ext jpg \
 --gpu_ids 0 1 2 3 4 5 6 7 \
 --checkpoint /home/goh/Documents/D3M/simclr_tf2_models/pretrained/r50_2x_sk0/saved_model/ \
 --model_dir /home/goh/Documents/D3M/mars_simclr_models \
 --keep_checkpoint_max 15 \
 --batch_size 128 \
 --learning_rate 0.2 \
 --resnet_depth 50 \
 --width_multiplier 2 \
 --sk_ratio 0.0
