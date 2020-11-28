#！/bin/sh
nohup python -u voter_nc_karate.py --cuda 0 --seed 327 --data_path '/data/chenmy/voter/seed1209ws1005000' >./log/ws100seed327.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 2 --seed 106 --data_path '/data/chenmy/voter/seed1027ws1005000' >./log/ws100seed106.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 3 --seed 259 --data_path '/data/chenmy/voter/seed1043ws1005000' >./log/ws100seed259.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 4 --seed 152 --data_path '/data/chenmy/voter/seed1051ws1005000' >./log/ws100seed152.txt 2>&1 &