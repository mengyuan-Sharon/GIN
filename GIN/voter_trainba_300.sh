nohup python -u voter_nc_bestrecord.py --cuda 1 >./log/voterseed135_ba_300_20_1.5w_emb64.txt 2>&1 &
nohup python -u voter_nc_bestrecord.py --cuda 3 --seed 2051 >./log/voterseed2051_ba_300_20_1.5w_emb64.txt 2>&1 &
nohup python -u voter_nc_bestrecord.py --cuda 5 --seed 2035 >./log/voterseed2035_ba_300_20_1.5w_emb64.txt 2>&1 &