import argparse
import glob
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import copy
import os
import matplotlib
import sys
import random
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import TrafficDataset, AttnLSTM, FSNet, TrafficTripleDataset, BertModel, FinalModel
from bert_util import BertTokenizer, AdamW

SEED = 2023
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='encrypted traffic classification')
parser.add_argument('--dataset', default=0, type=int, help='datasets for training (0-2)')
parser.add_argument('--method', default="Whisper", choices=["Whisper", "Characterize", "Robust", "Flowlens",
                                                            "ShortTerm",
                                                            "AttnLSTM", "Fs-net", "ETBert"],
                    help='method to use')
parser.add_argument('--noise', default="0.0", type=str, help='noise ratio')
parser.add_argument('--num_layer', default=2, type=int, help='Number of Layers')
parser.add_argument('--with_relu', default=1, type=int, help='use relu (1) or not (0)')
parser.add_argument('--with_ht', default=1, type=int, help='use ht (1) or not (0)')
parser.add_argument('--temp', default=0.1, type=float, help='temperature')
parser.add_argument('--with_re', default=1, type=int, help='use re (1) or not (0)')
parser.add_argument('--data_size', default=1.0, type=float, help='data size')
args = parser.parse_args()
print(args)
filenames = glob.glob(f"data_{args.noise}/0_SJTUAN21/*/{args.method}/*.*") \
            + glob.glob(f"data_{args.noise}/1_ISCXVPN/{args.method}/*.*") \
            + glob.glob(f"data_{args.noise}/2_ISCXTor/tor/{args.method}/*.*") \
            + glob.glob(f"data_{args.noise}/3_USTC-TFC/*/{args.method}/*.*") \
            + glob.glob(f"data_{args.noise}/5_Cross-Platform/*/*/{args.method}/*.*")
filenames = [filename for filename in filenames if filename.split("/")[1][0] == str(args.dataset)]
train_filenames = sorted([filename for filename in filenames if "train" in filename])
test_filenames = sorted([filename for filename in filenames if "test" in filename])

if args.dataset == 0:
    classifier_position = 1
else:
    classifier_position = 0

if args.dataset == 5:
    classifier = sorted(set([filename.split("/")[-4] for filename in filenames]))
else:
    classifier = sorted(set([filename.split("/")[-1].split("_")[classifier_position] for filename in filenames]))

if args.dataset == 3:
    classifier = sorted(set([c1.split("-")[0] for c1 in classifier]))


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if not os.path.exists("result"):
    os.mkdir("result")
if args.method == "ShortTerm":
    sys.stdout = Logger(
        f"result/Dataset_{args.dataset}_Method_{args.method}_Noise_{args.noise}_NumLayer_{args.num_layer}_ReLU_{args.with_relu}_HT_{args.with_ht}_RE_{args.with_re}_DataSize_{args.data_size}_Temp_{args.temp}.txt")
else:
    sys.stdout = Logger(f"result/Dataset_{args.dataset}_Method_{args.method}_Noise_{args.noise}.txt")


def load_data(file_names, classify):
    X = []
    Y = []
    for filename in tqdm(file_names):
        if args.dataset == 5:
            cls = filename.split("/")[-4]
        else:
            cls = filename.split("/")[-1].split("_")[classifier_position]
        if args.dataset == 3:
            cls = cls.split("-")[0]
        cls_number = classify.index(cls)
        data_arr = np.load(filename)
        X.append(data_arr)
        Y.extend([cls_number] * len(data_arr))
    X = np.concatenate(X, axis=0)
    Y = np.asarray(Y)
    X = np.nan_to_num(X)
    return X, Y


def load_data_csv(file_names, classify):
    X = []
    Y = []
    for filename in tqdm(file_names):
        if args.dataset == 5:
            cls = filename.split("/")[-4]
        else:
            cls = filename.split("/")[-1].split("_")[classifier_position]
        if args.dataset == 3:
            cls = cls.split("-")[0]
        cls_number = classify.index(cls)
        with open(filename, encoding='utf-8') as f:
            data_arr = np.loadtxt(f, str, delimiter=",")
            if len(data_arr.shape) == 1:
                continue
            data_arr = data_arr[1:, :].astype(np.int64)
        X.append(data_arr)
        Y.extend([cls_number] * len(data_arr))
    X = np.concatenate(X, axis=0)
    Y = np.asarray(Y)
    return X, Y


def load_data_bert(file_names, classify, tokenizer):
    src_ls = []
    cls_ls = []
    seg_ls = []
    for filename in tqdm(file_names):
        if args.dataset == 5:
            cls = filename.split("/")[-4]
        else:
            cls = filename.split("/")[-1].split("_")[classifier_position]
        if args.dataset == 3:
            cls = cls.split("-")[0]
        cls_number = classify.index(cls)
        with open(filename, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                src = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(line))
                seg = [1] * len(src)

                max_seq_length = 512
                if len(src) > max_seq_length:
                    src = src[: max_seq_length]
                    seg = seg[: max_seq_length]
                while len(src) < max_seq_length:
                    src.append(0)
                    seg.append(0)
                src_ls.append(src)
                cls_ls.append(cls_number)
                seg_ls.append(seg)

    src_ls = np.array(src_ls)
    seg_ls = np.array(seg_ls)
    cls_ls = np.array(cls_ls)
    return src_ls, seg_ls, cls_ls


if args.method in ["Whisper", "Characterize", "Robust", "Flowlens"]:
    if args.method == "Flowlens":
        train_X, train_Y = load_data_csv(train_filenames, classifier)
        test_X, test_Y = load_data_csv(test_filenames, classifier)
    else:
        train_X, train_Y = load_data(train_filenames, classifier)
        test_X, test_Y = load_data(test_filenames, classifier)

        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)
    print("train samples:", len(train_Y), np.bincount(train_Y).tolist())
    print("test samples:", len(test_Y), np.bincount(test_Y).tolist())

    if args.method == "Whisper":
        prototypes = []
        for i in range(len(classifier)):
            prototype = np.mean(train_X[train_Y == i], axis=0)
            prototypes.append(prototype)
        prototypes = np.asarray(prototypes)
        distances = cdist(test_X, prototypes, metric="euclidean")
        y_pred = np.argmin(distances, axis=-1)
    elif args.method == "Characterize":
        tree = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=1)
        tree.fit(train_X, train_Y)
        y_pred = tree.predict(test_X)
    elif args.method == "Robust":
        forest = RandomForestClassifier(criterion='gini', n_estimators=25, max_depth=8, random_state=1)
        forest.fit(train_X, train_Y)
        y_pred = forest.predict(test_X)
    elif args.method == "Flowlens":
        clf = MultinomialNB()
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)

    acc = round(accuracy_score(test_Y, y_pred), 4)
    precision = round(precision_score(test_Y, y_pred, average="weighted"), 4)
    recall = round(recall_score(test_Y, y_pred, average="weighted"), 4)
    f1 = round(f1_score(test_Y, y_pred, average="weighted"), 4)
    print(f"{args.method} on Dataset {args.dataset}\nAC: {acc}, PR: {precision}, RC: {recall}, F1: {f1}")
elif args.method in ["ShortTerm", "AttnLSTM", "Fs-net", "ETBert"]:
    if args.method == "ETBert":
        tokenizer = BertTokenizer()
        train_X, train_seg, train_Y = load_data_bert(train_filenames, classifier, tokenizer)
        test_X, test_seg, test_Y = load_data_bert(test_filenames, classifier, tokenizer)
    else:
        train_X, train_Y = load_data(train_filenames, classifier)
        test_X, test_Y = load_data(test_filenames, classifier)

    rng = np.random.RandomState(2023)
    inds = []
    for label in np.unique(train_Y):
        cls_inds = np.where(train_Y == label)[0]

        num = int(args.data_size * len(cls_inds))
        inds.extend(cls_inds[:num])
    inds = np.array(inds)
    train_X = train_X[inds]
    train_Y = train_Y[inds]

    print("train samples:", len(train_Y), np.bincount(train_Y).tolist())
    print("test samples:", len(test_Y), np.bincount(test_Y).tolist())

    if args.method == "AttnLSTM":
        train_ls = []
        for i in range(len(train_X)):
            p = 0  # random.randint(0, 80)
            train_ls.append(train_X[i, p: p + 20])
        train_X = np.array(train_ls)

        test_ls = []
        for i in range(len(test_X)):
            p = 0  # random.randint(0, 80)
            test_ls.append(test_X[i, p: p + 20])
        test_X = np.array(test_ls)

        train_X = train_X / 255.0
        test_X = test_X / 255.0

    if args.method == "ShortTerm":
        model = FinalModel(len(classifier), args.num_layer, args.with_relu, args.with_ht, args.with_re,
                           args.dataset, args.temp)
    elif args.method == "AttnLSTM":
        model = AttnLSTM(len(classifier))
    elif args.method == "Fs-net":
        model = FSNet(len(classifier))
    elif args.method == "ETBert":
        model = BertModel(len(classifier), len(tokenizer.vocab))
        msg = model.load_state_dict(torch.load("pretrained_model.bin",
                                               map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0',
                                                             'cuda:3': 'cuda:0'}),
                                    strict=False)
        print(msg)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if args.method == "ETBert":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            0.001,
            momentum=0.9,
            weight_decay=0.0003
        )

    if args.method == "ETBert":
        train_set = TrafficTripleDataset(train_X, train_seg, train_Y)
        test_set = TrafficTripleDataset(test_X, test_seg, test_Y)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False, drop_last=False, num_workers=0)
    else:
        train_set = TrafficDataset(train_X, train_Y)
        test_set = TrafficDataset(test_X, test_Y)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
        ema_model = copy.deepcopy(model)
        ema_model.cuda()

    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    test_f1_list = []
    if args.method == "ETBert":
        max_epoch = 4
    elif args.dataset == 2 and args.method != "ShortTerm":
        max_epoch = 100
    else:
        max_epoch = 50
    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        for step in tqdm(range(1000)):
            try:
                input, target = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input, target = next(train_iter)

            if args.method == "ETBert":
                input, seg = input
                seg = Variable(seg).cuda().to(torch.float32)

            input = Variable(input).cuda().to(torch.float32)
            target = Variable(target).cuda().to(torch.long)

            optimizer.zero_grad()
            if args.method == "Fs-net":
                logits, recon_logits = model(input)
                loss = criterion(logits, target)
                loss += criterion(recon_logits.view(-1, 10000), torch.clamp(input, max=9999).long().view(-1))
            elif args.method == "ShortTerm":
                logits = model(input)
                loss = criterion(logits, target)
            elif args.method == "ETBert":
                logits = model(input, seg)
                loss = criterion(logits, target)
            else:
                logits = model(input)
                loss = criterion(logits, target)
            train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            if args.method != "ETBert":
                for emp_p, p in zip(ema_model.parameters(), model.parameters()):
                    emp_p.data = 0.999 * emp_p.data + (1 - 0.999) * p.data
                for emp_p, p in zip(ema_model.buffers(), model.buffers()):
                    emp_p.data = 0.999 * emp_p.data + (1 - 0.999) * p.data

        train_loss /= 1000
        train_loss_list.append(train_loss)

        if args.method == "ETBert":
            test_model = model
        else:
            test_model = ema_model

        test_model.eval()
        with torch.no_grad():
            logits_softmax_ls = []
            target_ls = []
            test_loss = 0
            for _, (input, target) in enumerate(test_loader):
                if args.method == "ETBert":
                    input, seg = input
                    seg = Variable(seg).cuda().to(torch.float32)

                input = Variable(input).cuda().to(torch.float32)
                target = Variable(target).cuda().to(torch.long)

                if args.method == "Fs-net":
                    logits, _ = test_model(input)
                elif args.method == "ShortTerm":
                    logits = test_model(input)
                elif args.method == "ETBert":
                    logits = test_model(input, seg)
                else:
                    logits = test_model(input)
                loss = criterion(logits, target)
                test_loss += loss.item()
                logits_softmax = nn.Softmax(-1)(logits)
                logits_softmax_ls.append(logits_softmax)
                target_ls.append(target)

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)

            logits_softmax_ls = torch.cat(logits_softmax_ls, dim=0)
            target_ls = torch.cat(target_ls, dim=0)

            target_pred = torch.argmax(logits_softmax_ls, dim=-1)

            acc = round(accuracy_score(test_Y, target_pred.detach().cpu().numpy()), 4)
            precision = round(precision_score(test_Y, target_pred.detach().cpu().numpy(), average="weighted"), 4)
            recall = round(recall_score(test_Y, target_pred.detach().cpu().numpy(), average="weighted"), 4)
            f1 = round(f1_score(test_Y, target_pred.detach().cpu().numpy(), average="weighted"), 4)

            print(f"{args.method} on Dataset {args.dataset}\n at epoch {epoch}:\n"
                  f"train loss: {train_loss} test loss: {test_loss}\nAC: {acc}, PR: {precision}, RC: {recall}, F1: {f1}")
            test_acc_list.append(acc)
            test_f1_list.append(f1)

            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(train_loss_list, 'r', label='train_loss')
            ax1.plot(test_loss_list, 'y', label='test_loss')

            ax1.legend()
            ax2.plot(test_acc_list, 'b', label='test_accuracy')
            ax2.plot(test_f1_list, 'y', label='test_f1')
            ax2.legend()
            test_acc_array = np.array(test_acc_list)
            max_indx = np.argmax(test_acc_array)
            show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
            ax2.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
                         xy=(max_indx, test_acc_array[max_indx].item()))
            if not os.path.exists("result"):
                os.mkdir("result")
            fig.savefig(
                f"result/Dataset_{args.dataset}_Method_{args.method}_Noise_{args.noise}_NumLayer_{args.num_layer}_ReLU_{args.with_relu}_HT_{args.with_ht}_RE_{args.with_re}_DataSize_{args.data_size}_Temp_{args.temp}.png")
