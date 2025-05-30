python scripts/try_checkpoint_in_simpler.py --task google_robot_pick_horizontal_coke_can --checkpoint_path /cephfs/shared/llm/open_pi_0/fractal_beta_step29576_2024-12-29_13-10_42.pt --recording --use_bf16 --use_torch_compile

python src/model/vla/pizero.py --text_only --load_pretrained_weights --use_bf16


bash slurm/train_multi_gpu.sh


