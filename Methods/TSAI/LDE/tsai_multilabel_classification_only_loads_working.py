# -*- coding: utf-8 -*-
""" Adapted from Tsai_Example_01a_MultiLabel_TSClassification_Working.ipynb

Adapted by Everton L. de Aguiar from Doug Williams (https://github.com/williamsdoug) and Ignacio Oguiza (https://github.com/timeseriesAI/tsai).

contact: oguiza@timeseriesAI.co


"""
###############################################################
## Configurações gerais
train_mode = 'gpu'
learn_flag = 'True'

subset = [0, 1, 2]
normalized = 0
scenario = 4
choose_model = 1
dg_presence = [0, 1]


dict_subset = {0: 'Aggregated', 1: 'Individual', 2: 'All'}
dict_normalized = {0: '', 1: 'Normalized'}
dict_scenario = {1: 'SC1', 2: 'SC2', 3: 'SC3', 4: 'SC4', 5: 'SC5', 6: 'SC6'}
dict_models = {1: 'InceptionTime', 2: 'Sequencer', 3: 'TST', 4: 'TSiT'}
##################################################################3


if train_mode == 'cpu':
    ###############################################################3
    ## DESATIVANDO A GPU, JÁ QUE O MODELO NÃO CABE NELA
    import os

    # Define a variável de ambiente CUDA_VISIBLE_DEVICES como uma string vazia para desativar a GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ##############################################################



configs = {
    "N_GRIDS": 5,
    "SIGNAL_BASE_LENGTH": 12800,
    "N_CLASS": 5,
    "USE_NO_LOAD": False,
    "USE_HAND_AUGMENTATION": True,
    "MARGIN_RATIO": 0.15,
    "DATASET_PATH": "drive/MyDrive/Scattering_Novo/dataset_original/Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.9,
    "FOLDER_PATH": "/media/everton/Dados_SATA/Artigo_Dataset/Colab/ST-NILM/SC4_Classify_Loads_and_Inverter/trained_models/",
    "FOLDER_DATA_PATH": "/media/everton/Dados_SATA/Artigo_Dataset/Colab/ST-NILM/SC4_Classify_Loads_and_Inverter/trained_models/",
    "N_EPOCHS_TRAINING": 5000,
    "PERCENTUAL": [1],
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 5000,
    "SNRdb": None,  # Nível de ruído em db
    "LOSS": 'binary_crossentropy',  # 'bce_weighted_loss', 'focal_loss', "binary_crossentropy"
}

import sys

sys.path.append("/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/src")
sys.path.append("/media/everton/Dados_SATA/Artigo_Dataset/Colab/TSAI")

from LoadSegments import LoadSegments

loadSegments = LoadSegments(configs)

# Defines

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True, by_sample=False):
    "Computes accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    correct = (inp > thresh) == targ.bool()
    if by_sample:
        return (correct.float().mean(-1) == 1).float().mean()
    else:
        inp, targ = flatten_check(inp, targ)
        return correct.float().mean()


def precision_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes precision when `inp` and `targ` are the same size."

    inp, targ = flatten_check(inp, targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    precision = TP / (TP + FP)
    return precision


def recall_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes recall when `inp` and `targ` are the same size."

    inp, targ = flatten_check(inp, targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    FN = torch.logical_and(~correct, (targ == 1).bool()).sum()

    recall = TP / (TP + FN)
    return recall


def specificity_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes specificity (true negative rate) when `inp` and `targ` are the same size."

    inp, targ = flatten_check(inp, targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TN = torch.logical_and(correct, (targ == 0).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    specificity = TN / (TN + FP)
    return specificity


def balanced_accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes balanced accuracy when `inp` and `targ` are the same size."

    inp, targ = flatten_check(inp, targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    TN = torch.logical_and(correct, (targ == 0).bool()).sum()
    FN = torch.logical_and(~correct, (targ == 1).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    balanced_accuracy = (TPR + TNR) / 2
    return balanced_accuracy


def Fbeta_multi(inp, targ, beta=1.0, thresh=0.5, sigmoid=True):
    "Computes Fbeta when `inp` and `targ` are the same size."

    inp, targ = flatten_check(inp, targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    TN = torch.logical_and(correct, (targ == 0).bool()).sum()
    FN = torch.logical_and(~correct, (targ == 1).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    beta2 = beta * beta

    if precision + recall > 0:
        Fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    else:
        Fbeta = 0
    return Fbeta


def F1_multi(*args, **kwargs):
    return Fbeta_multi(*args, **kwargs)  # beta defaults to 1.0


for subsets_selector in subset:
    for inverter_presence in dg_presence:
        output_model_path = '/media/everton/Dados_SATA/Artigo_Dataset/Colab/TSAI/models/'
        output_model_path = output_model_path + dict_models[choose_model] + '/'

        from tsai.all import *

        my_setup()
        # device = torch.device('cpu') # Train on CPU

        if inverter_presence==1:
            file_name = 'Only_loads_with_inverter_'
        elif inverter_presence==0:
            file_name = 'Only_loads_without_inverter_'


        PATH = Path(output_model_path + dict_scenario[scenario] + '_' + dict_subset[subsets_selector] + dict_normalized[
            normalized] + '/' +  file_name + 'Multilabel.pkl')
        PATH.parent.mkdir(parents=True, exist_ok=True)

        train_x, train_y, test_x, test_y = loadSegments.load_segments(subsets_selector=subsets_selector,
                                                                      normalized=normalized)

        import numpy as np
        # all_x = np.append(train_x,test_x, axis=0)
        all_x = train_x
        # all_y = np.append(train_y,test_y, axis=0)
        all_y = train_y

        # Escolher os índices nos quais o inversor vale 'inverter_presence'

        train_index_list = np.where(all_y[:,0,4]==inverter_presence)
        train_index_list = train_index_list[0]

        test_index_list = np.where(test_y[:,0,4]==inverter_presence)
        test_index_list = test_index_list[0]

        test_y = test_y[:,:,:-1] # Excluding inverter from labels
        test_y = test_y[test_index_list]
        test_x = test_x[test_index_list]


        all_y = all_y[:,:,:-1] # Excluding inverter from labels
        all_y = all_y[train_index_list]
        all_x = all_x[train_index_list]


        # del train_x, train_y, test_x, test_y
        del train_x, train_y





        # subsets_selector
        # 0 - Aggregated
        # 1 - Individual
        # 2 - All




        def bin_to_multi(y_bin):
            y_multi = []  # empty list
            y = []
            loads_dict = {0: 'Iron', 1: 'Motor', 2: 'Driller', 3: 'Dimmer', 4: 'Inverter'}
            for k in range(y_bin.shape[0]):
                for load in range(y_bin.shape[1]):
                    if y_bin[k, load] == 1:
                        if len(y) == 0:
                            y = [loads_dict[load]]
                        else:
                            y.append(loads_dict[load])
                y_multi.append(y)
                y = []
            return y_multi


        # labeler = ReLabeler(class_map)
        # y_multi = labeler(y)
        y_multi = bin_to_multi(all_y[:, 0, :])
        X = all_x
        X = np.moveaxis(X, 1, 2)
        test_x = np.moveaxis(test_x, 1, 2)
        # label_counts = collections.Counter([a for r in y_multi for a in r])
        # print('Counts by label:', dict(label_counts))

        del all_x, all_y



        tfms = [None, TSMultiLabelClassification()]  # TSMultiLabelClassification() == [MultiCategorize(), OneHotEncode()]
        batch_tfms = TSStandardize()


        def construct_dataset_structure(X, y_multi, tfms, batch_tfms, train_index=None,
                                        val_index=None):  # train_index and val_index are the same list of index we used before
            train_index = train_index.astype(int)
            val_index = val_index.astype(int)
            if train_index.any() == None:
                # There is no validation subset
                splits = None
            elif train_index.shape[0] != 0:
                splits = [train_index, val_index]

            dls = get_ts_dls(X, y_multi, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
            return dls


        # Generating train and validation subsets indexes
        index = np.linspace(start=0, stop=len(y_multi) - 1, num=len(y_multi))  # 0, 1, 2, 3, ... , y_multi.shape[0]
        train_index = np.random.choice(index, round(0.8 * index.shape[0]), replace=False)
        val_index = np.random.choice(index, round(0.2 * index.shape[0]), replace=False)  # in fact, this is the test subset

        # dls = construct_dataset_structure(X, y_multi, tfms=tfms, batch_tfms=batch_tfms, train_index=train_index, val_index=val_index)
        dls = construct_dataset_structure(X, y_multi, tfms=tfms, batch_tfms=batch_tfms, train_index=train_index,
                                          val_index=val_index)

        dls.dataset



        label_counts = collections.Counter([a for r in y_multi for a in r])





        if learn_flag == 'True':
            metrics = [accuracy_multi, precision_multi, recall_multi, F1_multi]

            if dict_models[choose_model] == 'InceptionTime':
                learn = ts_learner(dls, InceptionTimePlus, metrics=metrics)
            elif dict_models[choose_model] == 'Sequencer':
                learn = ts_learner(dls, TSSequencerPlus, metrics=metrics)
            elif dict_models[choose_model] == 'TST':
                learn = ts_learner(dls, TSTPlus, metrics=metrics)
            elif dict_models[choose_model] == 'TSiT':
                learn = ts_learner(dls, TSiTPlus, metrics=metrics)
            else:
                print("Model does not exist.")


            learn.fit_one_cycle(10, lr_max=1e-3)
            learn.export(PATH)


        if train_mode == 'cpu':
            learn = load_learner(PATH, cpu=True)
        elif train_mode == 'gpu':
            learn = load_learner(PATH, cpu=False)



        splits = [train_index.astype(int), val_index.astype(int)]

        # cpu or gpu
        probas, _, preds = learn.get_X_preds(test_x)
        # probas, _, preds = learn_gpu.get_X_preds(X[splits[1]])

        preds = np.array(preds)
        probas = np.array(probas)

        Pred = {'probas': probas, 'preds': preds, 'train_index': train_index, 'val_index': val_index, 'test_y': test_y}
        import scipy.io as sio



        sio.savemat(
            output_model_path + dict_scenario[scenario] + '_' + dict_subset[subsets_selector] + '/' + dict_normalized[
                normalized] + file_name + 'Predictions.mat', Pred)
        import numpy as np

        np.save(output_model_path + dict_scenario[scenario] + '_' + dict_subset[subsets_selector] + '/' + dict_normalized[
            normalized] + file_name + 'test_predictions.npy', preds)
        np.save(output_model_path + dict_scenario[scenario] + '_' + dict_subset[subsets_selector] + '/' + dict_normalized[
            normalized] + file_name + 'test_y.npy', test_y)

