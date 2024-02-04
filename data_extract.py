# coding=utf-8
# 提取连续TCP包的特征：Payload（含TCP头部分，不包括端口和序列号）、Payload长度、时间间隔、TTL、IPFlag、TCPFlag
import glob
from scapy.all import PcapReader
import numpy as np
import binascii
from tqdm import tqdm
import os

filenames = glob.glob("traffic_data/0_SJTUAN21/*/*.cap") + glob.glob("traffic_data/4_CICIOT/5-Active/Active/*.pcap")


def extract(payload):
    dic = {payload.name: payload}
    payload = payload.payload
    while payload.name != "NoPayload":
        dic[payload.name] = payload
        payload = payload.payload
    return dic


for filename in tqdm(filenames):
    basename = filename.split("/")[-1].split(".")[0]
    new_dir = f"RawData" + "/".join(filename.split("traffic_data")[-1].split("/")[:-1])
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    with PcapReader(filename) as fdesc:
        length_sequence = []
        time_sequence = []
        ttl_sequence = []
        ip_flag_sequence = []
        tcp_flag_sequence = []
        packet_raw_string_sequence = []
        while True:
            try:
                packet = fdesc.read_packet()
                result = extract(packet)
                if "TCP" in result:
                    time = float(packet.time)
                    if result["TCP"].payload.name == "NoPayload":
                        length = 0
                    else:
                        length = len(result["TCP"].payload)
                    ttl = result["IP"].ttl
                    data = (binascii.hexlify(bytes(result["TCP"])))
                    packet_string = data.decode()[24:24+128*2+2]
                    ip_flag = result["IP"].flags.value
                    tcp_flag = result["TCP"].flags.value

                    time_sequence.append(time)
                    length_sequence.append(length)
                    packet_raw_string_sequence.append(packet_string)
                    ttl_sequence.append(ttl)
                    ip_flag_sequence.append(ip_flag)
                    tcp_flag_sequence.append(tcp_flag)
            except EOFError:
                break
    if len(time_sequence) > 0:
        time_sequence = np.array(time_sequence)
        time_sequence -= time_sequence[0]
        time_sequence = time_sequence[1:] - time_sequence[:-1]
        time_sequence = np.insert(time_sequence, 0, 0)

        length_sequence = np.array(length_sequence)
        packet_raw_string_sequence = np.array(packet_raw_string_sequence)
        ttl_sequence = np.array(ttl_sequence)
        ip_flag_sequence = np.array(ip_flag_sequence)
        tcp_flag_sequence = np.array(tcp_flag_sequence)

        np.save(new_dir + "/" + basename + "_L.npy", length_sequence)
        np.save(new_dir + "/" + basename + "_T.npy", time_sequence)
        np.save(new_dir + "/" + basename + "_P.npy", packet_raw_string_sequence)
        np.save(new_dir + "/" + basename + "_O.npy", ttl_sequence)
        np.save(new_dir + "/" + basename + "_F.npy", ip_flag_sequence)
        np.save(new_dir + "/" + basename + "_C.npy", tcp_flag_sequence)
