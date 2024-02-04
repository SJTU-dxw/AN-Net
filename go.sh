set -e

python -u data_extract.py
python -u add_noise.py
python -u data_process.py

python -u main.py --dataset 0 --noise 0.0 --method ShortTerm
python -u main.py --dataset 0 --noise 0.0 --method Whisper
python -u main.py --dataset 0 --noise 0.0 --method Characterize
python -u main.py --dataset 0 --noise 0.0 --method Robust
python -u main.py --dataset 0 --noise 0.0 --method Flowlens
python -u main.py --dataset 0 --noise 0.0 --method AttnLSTM
python -u main.py --dataset 0 --noise 0.0 --method Fs-net
python -u main.py --dataset 0 --noise 0.0 --method ETBert

python -u main.py --dataset 0 --noise 0.5_TLS --method ShortTerm
python -u main.py --dataset 0 --noise 0.5_TLS --method Whisper
python -u main.py --dataset 0 --noise 0.5_TLS --method Characterize
python -u main.py --dataset 0 --noise 0.5_TLS --method Robust
python -u main.py --dataset 0 --noise 0.5_TLS --method Flowlens
python -u main.py --dataset 0 --noise 0.5_TLS --method AttnLSTM
python -u main.py --dataset 0 --noise 0.5_TLS --method Fs-net
python -u main.py --dataset 0 --noise 0.5_TLS --method ETBert

python -u main.py --dataset 0 --noise 0.5_SIM --method ShortTerm
python -u main.py --dataset 0 --noise 0.5_SIM --method Whisper
python -u main.py --dataset 0 --noise 0.5_SIM --method Characterize
python -u main.py --dataset 0 --noise 0.5_SIM --method Robust
python -u main.py --dataset 0 --noise 0.5_SIM --method Flowlens
python -u main.py --dataset 0 --noise 0.5_SIM --method AttnLSTM
python -u main.py --dataset 0 --noise 0.5_SIM --method Fs-net
python -u main.py --dataset 0 --noise 0.5_SIM --method ETBert

python -u main.py --dataset 0 --noise 0.75_TLS --method ShortTerm
python -u main.py --dataset 0 --noise 0.75_TLS --method Whisper
python -u main.py --dataset 0 --noise 0.75_TLS --method Characterize
python -u main.py --dataset 0 --noise 0.75_TLS --method Robust
python -u main.py --dataset 0 --noise 0.75_TLS --method Flowlens
python -u main.py --dataset 0 --noise 0.75_TLS --method AttnLSTM
python -u main.py --dataset 0 --noise 0.75_TLS --method Fs-net
python -u main.py --dataset 0 --noise 0.75_TLS --method ETBert

python -u main.py --dataset 0 --noise 0.75_SIM --method ShortTerm
python -u main.py --dataset 0 --noise 0.75_SIM --method Whisper
python -u main.py --dataset 0 --noise 0.75_SIM --method Characterize
python -u main.py --dataset 0 --noise 0.75_SIM --method Robust
python -u main.py --dataset 0 --noise 0.75_SIM --method Flowlens
python -u main.py --dataset 0 --noise 0.75_SIM --method AttnLSTM
python -u main.py --dataset 0 --noise 0.75_SIM --method Fs-net
python -u main.py --dataset 0 --noise 0.75_SIM --method ETBert