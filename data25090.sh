# 传递到5090(ssh)
rsync -avz --progress data/openpi/franka_droid_lerobot_2_8/ \
    sustechdl@MS-7E06:/home/sustechdl/three/openpi_fintune/franka_droid_lerobot_2_8/

# 传到4090(ssh)
rsync -avz -e 'ssh -p 2222' --progress data/openpi/franka_lerobot_4_9_audio/ three@192.168.5.55:~/code/openpi/franka_lerobot_5_1_audio/