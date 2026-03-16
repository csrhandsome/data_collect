# 传递到5090(ssh)
rsync -avz --progress data/openpi/franka_droid_lerobot_2_8/ \
    sustechdl@MS-7E06:/home/sustechdl/three/openpi_fintune/franka_droid_lerobot_2_8/

# 传到4090(ssh)
rsync -avz -e 'ssh -p 2222' --progress data/openpi/franka_franka_lerobot_3_11/ three@192.168.5.38:~/code/openpi/franka_droid_lerobot_3_11/