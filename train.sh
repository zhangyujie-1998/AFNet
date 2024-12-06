# WPC training
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py \
--save_flag True \
--num_epochs 50 \
--batch_size 8 \
--test_patch_num 10 \
--learning_rate 0.00002 \
--decay_rate 5e-4 \
--database WPC \
--data_dir_texture ./database/WPC/proj_6view_512_texture \
--data_dir_depth ./database/WPC/proj_6view_512_depth \
--data_dir_mask ./database/WPC/proj_6view_512_mask \
--output_dir ./results/WPC/ \
--k_fold_num 5 \
> logs/log_WPC.txt 2>&1 &

# SJTU training
CUDA_VISIBLE_DEVICES=4 nohup python -u train.py \
--save_flag True \
--num_epochs 50 \
--batch_size 8 \
--test_patch_num 10 \
--learning_rate 0.00002 \
--decay_rate 5e-4 \
--database SJTU \
--data_dir_texture ./database/SJTU-PCQA/proj_6view_512_texture \
--data_dir_depth ./database/SJTU-PCQA/proj_6view_512_depth \
--data_dir_mask ./database/SJTU-PCQA/proj_6view_512_mask \
--output_dir ./results/SJTU/ \
--k_fold_num 9 \
> logs/log_SJTU.txt 2>&1 &

# LSPCQA training
CUDA_VISIBLE_DEVICES=5 nohup python -u train.py \
--save_flag True \
--num_epochs 50 \
--batch_size 8 \
--test_patch_num 10 \
--learning_rate 0.00002 \
--decay_rate 5e-4 \
--database LSPCQA \
--data_dir_texture ./database/LSPCQA/proj_6view_512_texture \
--data_dir_depth ./database/LSPCQA/proj_6view_512_depth \
--data_dir_mask ./database/LSPCQA/proj_6view_512_mask \
--output_dir ./results/LSPCQA/ \
--k_fold_num 5 \
> logs/log_LSPCQA.txt 2>&1 &