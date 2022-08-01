# -*- coding: utf-8 -*-
import json
import argparse

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DataPrecessForSentence
from utils import train, validate, eval_object, my_plot
# from transformers import BertTokenizer, AutoTokenizer, DataCollatorForSeq2Seq
from model import get_model
from transformers.optimization import AdamW
from config import model_dict



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert', type=str, required=False, help='使用什么模型')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练多少代')
    # parser.add_argument('--dir_name', default='xinwen', type=str, required=False, help='训练集存放目录,里面包含train.csv test.csv dev.csv')
    parser.add_argument('--dir_name', default='cner', type=str, required=False, help='训练集存放目录,里面包含train.csv test.csv dev.csv')
    parser.add_argument('--max_seq_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='训练的batch size')
    # parser.add_argument('--max_gradient_norm', default=16, type=int, required=False, help='训练的batch size')
    parser.add_argument('--read_n_num', default=None, type=int, required=False, help='读取train.csv行数，不输入为全部读取')
    parser.add_argument('--checkpoint', default=None, type=str, required=False, help='训练结果存放位置, 有这个参数则从checkpoint继续训练')
    parser.add_argument('--freeze_bert_head', default=False, help='是否冻结前部分bert参数')
    parser.add_argument('--use_sample', default=False, help='是否过采样')
    parser.add_argument('--text_col_name', default='raw_sen', type=str, required=False, help='train.csv文本列名字')
    parser.add_argument('--class_col_name', default='label', type=str, required=False, help='train.csv标签列名字')
    parser.add_argument('--csv_sep', default=',', type=str, required=False, help='csv列间隔')
    parser.add_argument('--csv_encoding', default='utf-8', type=str, required=False, help='csv编码格式')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--patience', default=40, type=float, required=False, help='early stop代数')
    parser.add_argument('--max_grad_norm', default=10, type=float, required=False, help='梯度修剪')
    args = parser.parse_args()
    return args

def init(args):
    """
    增加其他参数，以及创建文件夹
    """
    target_file = f'models/{args.dir_name}/{args.model}_best.pth.tar'  # 模型存储路径
    pretrain_dir = f'./models/{args.dir_name}/{args.model}/'  # 模型存储路径,存储了两种类型，一种是torch.save 一种是model.from_pretrained
    test_pred_out = f"data/{args.dir_name}/test_data_predict.csv"
    train_file = f"data/{args.dir_name}/train.csv"
    dev_file = f"data/{args.dir_name}/dev.csv"
    test_file = f"data/{args.dir_name}/test.csv"
    # json_dict = f"data/{args.dir_name}/class.txt"
    json_dict = f"data/{args.dir_name}/label_2_id.json"
    data_info_file = f"data/{args.dir_name}/label_count.png"

    with open(json_dict, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    # label2id = {label.strip(): i for i, label in enumerate(classes)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    print(f"num_labels 是{num_labels}")

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.pretrain_dir = pretrain_dir
    args.target_file = target_file
    args.test_pred_out = test_pred_out
    args.train_file = train_file
    args.dev_file = dev_file
    args.test_file = test_file
    args.id2label = id2label
    args.label2id = label2id
    args.num_labels = num_labels

    # 自动创建文件夹
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def main():
    args = set_args()
    init(args)


    Tokenizer = eval_object(model_dict[args.model][0])
    bert_path_or_name = model_dict[args.model][-1]
    tokenizer = Tokenizer.from_pretrained(bert_path_or_name)
    # print(tokenizer.return_offsets_mapping)

    print("\t* Loading training data...")
    processed_datasets_train = DataPrecessForSentence(tokenizer, args.train_file, args)
    processed_datasets_dev = DataPrecessForSentence(tokenizer, args.dev_file, args)
    print("\t* Loading validation data...")
    # dev_data = DataPrecessForSentence(tokenizer, dev_file)
    # dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = get_model(args)
    train_loader = DataLoader(processed_datasets_train, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(processed_datasets_dev, shuffle=True, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    bert_lr = 3e-5
    crf_lr = 1e-3
    linear_lr = 1e-3
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': bert_lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': bert_lr},
        ]

    # optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=bert_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_f1_list, dev_f1_list = [], []
    # Continuing training from a checkpoint if one was given as argument
    if args.checkpoint:
        print(f'载入checkpoint文件{args.checkpoint}')
        checkpoint_save = torch.load(args.checkpoint)
        # print(checkpoint_save.keys())
        start_epoch = checkpoint_save["epoch"] + 1
        # start_epoch = 1
        best_score = checkpoint_save["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint_save["model"])
        # optimizer.load_state_dict(checkpoint_save["optimizer"])
        epochs_count = checkpoint_save["epochs_count"]
        train_losses = checkpoint_save["train_losses"]
        valid_losses = checkpoint_save["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    epoch_time, epoch_loss, f1_score = validate(model, dev_loader, args)
    print("before train-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
          .format(epoch_time, epoch_loss, (f1_score * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training model on device: {}".format(args.device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        # print(model)
        # epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        epoch_time, epoch_loss = train(model, train_loader, optimizer, args)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: "
              .format(epoch_time, epoch_loss))
        print("* Validation for epoch {}:".format(epoch))
        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader)

        train_epoch_time, train_epoch_loss, train_f1_score = validate(model, train_loader, args)
        epoch_time, epoch_loss, f1_score = validate(model, dev_loader, args)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, f1_score: {:.4f}% \n"
              .format(epoch_time, epoch_loss, (f1_score * 100)))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(f1_score)
        # Early stopping on validation accuracy.
        if f1_score < best_score:
            patience_counter += 1
        else:
            print('save model')
            best_score = f1_score
            patience_counter = 0

            tokenizer.save_pretrained(args.pretrain_dir)
            model.save_pretrained(args.pretrain_dir)

            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        # "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       args.target_file
                       )
        dev_f1_list.append(f1_score)
        train_f1_list.append(train_f1_score)
        my_plot(train_f1_list, dev_f1_list, train_losses, args)
        if patience_counter >= args.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    # train_file = "../data/TextClassification/mnli/train.csv"
    # df = pd.read_csv(train_file, engine='python', error_bad_lines=False)

    main()
