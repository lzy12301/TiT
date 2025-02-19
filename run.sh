CUDA_VISIBLE_DEVICES=3 python train_dit.py --data kol --version test \
    --patch_size 1 2 2 --hidden_size 384 --depth 6 --mlp_ratio 4 --num_heads 6 --in_channels 1 \
    --dtype float16 --enable_flash_attn 1 --crop_size 32 --num_frames 16 \
    --batch_size 40 --small_batch_size 40 --lr 5e-5 --n_iters 120000 \
    --sigma_data 1 --is_scalar 1 --data_loc data/kf_2d_re1000_32_40seed_bthwc_train.npy
