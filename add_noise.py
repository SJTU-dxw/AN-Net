# coding=utf-8
# 插入噪声，策略：插入TLS；模拟包，利用高斯分布
import glob
import numpy as np
from tqdm import tqdm
import random
import os

noises = [0.5, 0.75]
SEED = 2023
MAX_NUM = 5000000
np.random.seed(SEED)
random.seed(SEED)

TLS_filenames = glob.glob("RawData/4_CICIOT/5-Active/Active/*.npy")
filenames = glob.glob("RawData/0_SJTUAN21/*/*.npy")

filenames = [filename[:-6] for filename in filenames]
TLS_filenames = [filename[:-6] for filename in TLS_filenames]
filenames = sorted(set(filenames))
TLS_filenames = sorted(set(TLS_filenames))


# 获得TLS所有包
TLS_time_sequence = []
TLS_length_sequence = []
TLS_ttl_sequence = []
TLS_ip_flag_sequence = []
TLS_tcp_flag_sequence = []
TLS_packet_raw_string_sequence = []
for filename in tqdm(TLS_filenames, desc="Loading TLS Pcap"):
    TLS_time_sequence.append(np.load(filename + "_T.npy"))
    TLS_length_sequence.append(np.load(filename + "_L.npy"))
    TLS_ttl_sequence.append(np.load(filename + "_O.npy"))
    TLS_ip_flag_sequence.append(np.load(filename + "_F.npy"))
    TLS_tcp_flag_sequence.append(np.load(filename + "_C.npy"))
    TLS_packet_raw_string_sequence.append(np.load(filename + "_P.npy"))
TLS_time_sequence = np.concatenate(TLS_time_sequence, axis=0)[:MAX_NUM]
TLS_length_sequence = np.concatenate(TLS_length_sequence, axis=0)[:MAX_NUM]
TLS_ttl_sequence = np.concatenate(TLS_ttl_sequence, axis=0)[:MAX_NUM]
TLS_ip_flag_sequence = np.concatenate(TLS_ip_flag_sequence, axis=0)[:MAX_NUM]
TLS_tcp_flag_sequence = np.concatenate(TLS_tcp_flag_sequence, axis=0)[:MAX_NUM]
TLS_packet_raw_string_sequence = np.concatenate(TLS_packet_raw_string_sequence, axis=0)[:MAX_NUM]

for noise in noises:
    for filename in tqdm(filenames):
        time_sequence = np.load(filename + "_T.npy")
        length_sequence = np.load(filename + "_L.npy")
        ttl_sequence = np.load(filename + "_O.npy")
        packet_raw_string_sequence = np.load(filename + "_P.npy")
        ip_flag_sequence = np.load(filename + "_F.npy")
        tcp_flag_sequence = np.load(filename + "_C.npy")

        alternative_num = int(100 * noise)
        instance_num = len(time_sequence) // 100
        for k in range(instance_num):
            index = np.arange(len(TLS_packet_raw_string_sequence) - alternative_num)
            tmp_index = np.random.choice(index, 1, replace=False)[0]

            r_index = np.arange(100 - alternative_num)
            s_index = np.random.choice(r_index, 1, replace=False)[0]

            start = k * 100 + s_index
            end = k * 100 + s_index + alternative_num

            time_sequence[start: end] = TLS_time_sequence[tmp_index: tmp_index + alternative_num]
            length_sequence[start: end] = TLS_length_sequence[tmp_index: tmp_index + alternative_num]
            ttl_sequence[start: end] = TLS_ttl_sequence[tmp_index: tmp_index + alternative_num]
            packet_raw_string_sequence[start: end] = TLS_packet_raw_string_sequence[tmp_index: tmp_index + alternative_num]
            ip_flag_sequence[start: end] = TLS_ip_flag_sequence[tmp_index: tmp_index + alternative_num]
            tcp_flag_sequence[start: end] = TLS_tcp_flag_sequence[tmp_index: tmp_index + alternative_num]

        basename = filename.split("/")[-1].split(".")[0]
        new_dir = f"RawData_{noise}_TLS" + "/".join(filename.split("RawData")[-1].split("/")[:-1])
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        np.save(new_dir + "/" + basename + "_L.npy", length_sequence)
        np.save(new_dir + "/" + basename + "_T.npy", time_sequence)
        np.save(new_dir + "/" + basename + "_P.npy", packet_raw_string_sequence)
        np.save(new_dir + "/" + basename + "_O.npy", ttl_sequence)
        np.save(new_dir + "/" + basename + "_F.npy", ip_flag_sequence)
        np.save(new_dir + "/" + basename + "_C.npy", tcp_flag_sequence)


# 模拟各个字段
def obtain_statistic(ls):
    return np.mean(ls), np.std(ls)


def obtain_sim(files):
    current_time_sequence = []
    current_length_sequence = []
    current_ttl_sequence = []
    current_ip_flag_sequence = []
    current_tcp_flag_sequence = []
    for filename in tqdm(files):
        current_time_sequence.append(np.load(filename + "_T.npy"))
        current_length_sequence.append(np.load(filename + "_L.npy"))
        current_ttl_sequence.append(np.load(filename + "_O.npy"))
        current_ip_flag_sequence.append(np.load(filename + "_F.npy"))
        current_tcp_flag_sequence.append(np.load(filename + "_C.npy"))
    current_time_sequence = np.concatenate(current_time_sequence, axis=0)
    current_length_sequence = np.concatenate(current_length_sequence, axis=0)
    current_ttl_sequence = np.concatenate(current_ttl_sequence, axis=0)
    current_ip_flag_sequence = np.concatenate(current_ip_flag_sequence, axis=0)
    current_tcp_flag_sequence = np.concatenate(current_tcp_flag_sequence, axis=0)
    dic = {
        "time": obtain_statistic(current_time_sequence),
        "length": obtain_statistic(current_length_sequence),
        "ttl": obtain_statistic(current_ttl_sequence),
        "ip": obtain_statistic(current_ip_flag_sequence),
        "tcp": obtain_statistic(current_tcp_flag_sequence),
        "num": len(current_time_sequence)
    }
    return dic


# 模拟正态分布
sim_dict = {}
dataset_list = sorted(set([filename.split("/")[1][0] for filename in filenames]))
for dset in dataset_list:
    tmp_filenames = [filename for filename in filenames if filename.split("/")[1][0] == dset]
    sim_dict[dset] = obtain_sim(tmp_filenames)

# 获得TLS所有包Payload，其他字段用以模拟
TLS_packet_raw_string_sequence = []
for filename in tqdm(TLS_filenames, desc="Loading TLS Pcap"):
    TLS_packet_raw_string_sequence.append(np.load(filename + "_P.npy"))
TLS_packet_raw_string_sequence = np.concatenate(TLS_packet_raw_string_sequence, axis=0)[:MAX_NUM]

for noise in noises:
    for filename in tqdm(filenames):
        time_sequence = np.load(filename + "_T.npy")
        length_sequence = np.load(filename + "_L.npy")
        ttl_sequence = np.load(filename + "_O.npy")
        packet_raw_string_sequence = np.load(filename + "_P.npy")
        ip_flag_sequence = np.load(filename + "_F.npy")
        tcp_flag_sequence = np.load(filename + "_C.npy")

        dset = filename.split("/")[1][0]
        alternative_num = int(100 * noise)
        instance_num = len(time_sequence) // 100
        for k in range(instance_num):
            index = np.arange(len(TLS_packet_raw_string_sequence) - alternative_num)
            tmp_index = np.random.choice(index, 1, replace=False)[0]

            r_index = np.arange(100 - alternative_num)
            s_index = np.random.choice(r_index, 1, replace=False)[0]

            start = k * 100 + s_index
            end = k * 100 + s_index + alternative_num

            time_sequence[start: end] = np.random.normal(loc=sim_dict[dset]["time"][0], scale=sim_dict[dset]["time"][1], size=alternative_num)
            length_sequence[start: end] = np.random.normal(loc=sim_dict[dset]["length"][0], scale=sim_dict[dset]["length"][1], size=alternative_num)
            ttl_sequence[start: end] = np.random.normal(loc=sim_dict[dset]["ttl"][0], scale=sim_dict[dset]["ttl"][1], size=alternative_num)
            packet_raw_string_sequence[start: end] = TLS_packet_raw_string_sequence[tmp_index: tmp_index + alternative_num]
            ip_flag_sequence[start: end] = np.random.normal(loc=sim_dict[dset]["ip"][0], scale=sim_dict[dset]["ip"][1], size=alternative_num)
            tcp_flag_sequence[start: end] = np.random.normal(loc=sim_dict[dset]["tcp"][0], scale=sim_dict[dset]["tcp"][1], size=alternative_num)

        basename = filename.split("/")[-1].split(".")[0]
        new_dir = f"RawData_{noise}_SIM" + "/".join(filename.split("RawData")[-1].split("/")[:-1])
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        np.save(new_dir + "/" + basename + "_L.npy", length_sequence)
        np.save(new_dir + "/" + basename + "_T.npy", time_sequence)
        np.save(new_dir + "/" + basename + "_P.npy", packet_raw_string_sequence)
        np.save(new_dir + "/" + basename + "_O.npy", ttl_sequence)
        np.save(new_dir + "/" + basename + "_F.npy", ip_flag_sequence)
        np.save(new_dir + "/" + basename + "_C.npy", tcp_flag_sequence)
