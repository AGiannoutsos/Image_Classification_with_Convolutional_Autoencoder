import json
import sys

data = {}
# data['model_info'] = {
#         "encoder_layers" : [["conv", 32, (3,3)],
#                             ["batchNorm"],
#                             ["conv", 32, (3,3)],
#                             ["pool", (2,2)],
#                             ["conv", 64, (3,3)],
#                             ["batchNorm"],
#                             ["conv", 64, (3,3)],
#                             ["pool", (2,2)],
#                             ["conv", 128, (3,3)],
#                             ["batchNorm"]]
#         ,
#         "decoder_layers" :  [["conv", 128, (3,3)],
#                             ["batchNorm"],
#                             ["conv", 64, (3,3)],
#                             ["batchNorm"],
#                             ["conv", 64, (3,3)],
#                             ["batchNorm"],
#                             ["upSample", (2,2)],
#                             ["conv", 32, (3,3)],
#                             ["batchNorm"],
#                             ["conv", 32, (3,3)],
#                             ["batchNorm"],
#                             ["upSample", (2,2)]]
#         ,
#         "activation_function": "Sigmoid"
#         ,
#         "batch_size":       32
#         ,
#         "epochs":           1
#         ,
#         "optimizer" :       ["adam", 0.01]
# }

data['model_info'] = {
    "dense_layers": [["dense", 50],
                    ["dense", 20]]
    ,
    "encoder_layers" : "here.h5"
    ,
    "batch_size":   32
    ,
    "dense_only_train_epochs": 1
    ,
    "full_train_epochs": 1
    ,
    "optimizer" :   ["adam", 0.001]
}

with open('config_classifier.txt', 'w') as out:
    json.dump(data, out)
# with open('config.txt') as json_file:
#     data = json.load(json_file)
#     print(data)
#     model_info = data['model_info']
#     print(type(model_info))
#     print(model_info)
#     # print(data)