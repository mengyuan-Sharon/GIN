nohup python -u test_sigmoid.py --cuda 1 --data_path '/data/chenmy/cml/2000cmlWS-100-10000.pickle' >./log/seed2000cmlws100-10.txt 2>&1 &
nohup python -u test_sigmoid.py --cuda 3 --data_path '/data/chenmy/cml/2010cmlWS-100-10000.pickle' >./log/seed2100cmlws100-10.txt 2>&1 &
nohup python -u test_sigmoid.py --cuda 4 --data_path '/data/chenmy/cml/527cmlWS-100-10000.pickle' >./log/seed527cmlws100-10.txt 2>&1 &
nohup python -u test_sigmoid.py --cuda 5 --data_path '/data/chenmy/cml/927cmlWS-100-10000.pickle' >./log/seed927cmlws100-10.txt 2>&1 &