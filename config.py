# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''
model_dict = {
    'bert': ('transformers.BertTokenizerFast',
             'transformers.BertForTokenClassification',
             'transformers.BertConfig',
             'bert-base-chinese'  # 使用模型
             ),
    # 'ernie': (
    #     'transformers.BertTokenizerFast',
    #     'transformers.BertModel',
    #     'transformers.AutoConfig',
    #     "nghuyong/ernie-1.0",  # 使用模型参数
    # ),
    'roberta': (
        'transformers.BertTokenizerFast',
        'transformers.RobertaForTokenClassification',
        'transformers.RobertaConfig',
        'hfl/chinese-roberta-wwm-ext',
    ),
    'albert': ('transformers.BertTokenizerFast',
               'transformers.AlbertForTokenClassification',
               'transformers.AutoConfig',
               "voidful/albert_chinese_tiny",  # 使用模型参数
               ),
}
# MODEL = 'roberta'
# MODEL = 'ernie'
# MODEL = 'albert'
MODEL = 'bert'
