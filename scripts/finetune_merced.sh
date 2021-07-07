python /home/goh/Documents/D3M/simclr-experiments/simclr/finetune.py \
 /home/goh/Documents/D3M/UCMerced_LandUse_PNG/Images \
 --dataset uc_merced_full \
 --ext png \
 --gpu_ids 0 1 2 3 4 5 6 7 \
 --checkpoint /home/goh/Documents/D3M/simclr_tf2_models/pretrained/r50_2x_sk1/saved_model/ \
 --model_dir /home/goh/Documents/D3M/merced_models \
 --keep_checkpoint_max 10 \
 --batch_size 64 \
 --learning_rate 0.16 \
 --resnet_depth 50 \
 --width_multiplier 2 \
 --sk_ratio 0.0625
