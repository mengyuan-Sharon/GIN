#！/bin/sh
nohup python -u voter_nc_karate.py --cuda 5 --seed 327 --data_path '/data/chenmy/voter/seed1043ws30015000' >./log/ws300seed327.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 2 --seed 106 --data_path '/data/chenmy/voter/seed1027ws30015000' >./log/ws300seed106.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 3 --seed 259 --data_path '/data/chenmy/voter/seed1051ws30015000' >./log/ws300seed259.txt 2>&1 & 
nohup python -u voter_nc_karate.py --cuda 4 --seed 152 --data_path '/data/chenmy/voter/seed1209ws30015000' >./log/ws300seed152.txt 2>&1 &
