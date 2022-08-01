# -*- coding: utf-8 -*-
from utils import eval_object
from config import model_dict


def get_model(args):
    ClassifyClass = eval_object(model_dict[args.model][1])
    ClassifyConfig = eval_object(model_dict[args.model][2])
    bert_path_or_name = model_dict[args.model][-1]

    config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=args.num_labels,
                                            label2id=args.label2id,
                                            id2label=args.id2label)
    model = ClassifyClass.from_pretrained(bert_path_or_name, config=config)  # /bert_pretrain/
    model = model.to(args.device)
    return model
