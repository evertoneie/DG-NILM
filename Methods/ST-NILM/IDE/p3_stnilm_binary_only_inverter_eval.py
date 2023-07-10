# -*- coding: utf-8 -*-
"""P3_1_ST-NILM_GD_NILM_Only_Inverter_Eval.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sWYgp-WQ0DxZDUYlIxanFiiopKuGtUvH

## Pre-Definitions and importing
"""

# root_path = "/content/drive/MyDrive/Doutorado/Artigo_Dataset/Colab/Multi_Label_GD_NILM"
# root_path = "/media/everton/Dados_SATA/Artigo_Dataset/Colab/LSTM_TR/LSTM_TR_SRC"


source_models_folder = './ST-NILM/IDE/trained_models/'

J = 4
Q = 8
choose_folder_flag = True


from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
import sys

sys.path.append("./src")
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

configs = {
    "N_GRIDS": 5,
    "SIGNAL_BASE_LENGTH": 512,
    "N_CLASS": 1,
    "USE_NO_LOAD": False,
    "USE_HAND_AUGMENTATION": True,
    "MARGIN_RATIO": 0.15,
    "DATASET_PATH": "drive/MyDrive/Scattering_Novo/dataset_original/Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.9,
    "FOLDER_PATH": "/media/everton/Dados_SATA/Artigo_Dataset/Colab/ST-NILM/SC3_Classify_Only_Inverter/trained_models/",
    "FOLDER_DATA_PATH": "/media/everton/Dados_SATA/Artigo_Dataset/Colab/ST-NILM/SC3_Classify_Only_Inverter/trained_models/",
    "N_EPOCHS_TRAINING": 5000,
    "PERCENTUAL": [1],
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 5000,
    "SNRdb": None,  # Nível de ruído em db
    "LOSS": 'binary_crossentropy',  # 'bce_weighted_loss', 'focal_loss', "binary_crossentropy", "binary_crossentropy"
}

# Custom losses list:
# 'Scattering1D': Scattering1D, \
# 'sumSquaredError': ModelHandler.sumSquaredError, \
# 'loss': ModelHandler.weighted_categorical_crossentropy(type_weights), \
# 'focal_loss': ModelHandler.multi_label_focal_loss(), \
# 'dice_loss': ModelHandler.dice_loss_multiclass(), \
# 'generalized_dice_loss': ModelHandler.generalized_dice_loss(), \
# 'jaccard_loss': ModelHandler.jaccard_loss_multiclass(), \
# 'bce_weighted_loss': ModelHandler.get_bce_weighted_loss(None)

ngrids = configs["N_GRIDS"]
signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
trainSize = configs["TRAIN_SIZE"]
folderDataPath = configs["FOLDER_DATA_PATH"]
folderPath = configs["FOLDER_PATH"]

dataHandler = DataHandler(configs)

"""#### Load Data"""

root_path = "/media/everton/Dados_SATA/Artigo_Dataset/Colab/ST-NILM"
#data_path = "/media/everton/Dados_SATA/Artigo_Dataset/Colab/LSTM_TR/LSTM_TR_SRC/segments"
data_path = "/media/everton/Dados_SATA/Artigo_Dataset/Colab/Multi_Label_GD_NILM"
segments_path = "/media/everton/Dados_SATA/Artigo_Dataset/Colab/LSTM_TR/LSTM_TR_SRC/segments"

# normalized = 0
# subsets_selector = 2
win_size = 512
# subsets_selector
# 0 - Aggregated
# 1 - Individual
# 2 - All

sys.path.append(root_path)

normalized = 0
subsets_selector = 2

# subsets_selector
# 0 - Aggregated
# 1 - Individual
# 2 - All with and without inverter
# 3 - All with inverter
# 4 - All without inverter
# 5 - individual with inverter
# 6 - individual without inverter
# 7 - aggregated with inverter
# 8 - aggregated without inverter


reduced_dataset_flag = False 
load_existing_segments = False

if subsets_selector == 0:
    complemento = "Detect_Only_Inverter_aggregated_512"
    complemento2 = "_aggregated"
elif subsets_selector == 1:
    complemento = "Detect_Only_Inverter_individual_512"
    complemento2 = "_individual"
elif subsets_selector == 2:
    complemento = "Detect_Only_Inverter_all_512"
    complemento2 = "_all"
elif subsets_selector == 3:
    complemento = "Detect_Only_Inverter_all_with_inverter_512"
    complemento2 = "_all_with_inverter"
elif subsets_selector == 4:
    complemento = "Detect_Only_Inverter_all_without_inverter_512"
    complemento2 = "_all_without_inverter"
elif subsets_selector == 5:
    complemento = "Detect_Only_Inverter_individual_with_inverter_512"
    complemento2 = "_individual_with_inverter"
elif subsets_selector == 6:
    complemento = "Detect_Only_Inverter_individual_without_inverter_512"
    complemento2 = "_individual_without_inverter"
elif subsets_selector == 7:
    complemento = "Detect_Only_Inverter_aggregated_with_inverter_512"
    complemento2 = "_aggregated_with_inverter"
elif subsets_selector == 8:
    complemento = "Detect_Only_Inverter_aggregated_without_inverter_512"
    complemento2 = "_aggregated_without_inverter"

if reduced_dataset_flag:
    complemento = "reduced_" + complemento

if normalized:
    complemento = "normalized_" + complemento


complemento3 = ""
complemento2 = complemento2 + '_' + configs["LOSS"]
segments_path = segments_path + "/" + complemento
segments_file_name = "segments.mat"

print(segments_path)


def remove_if_exists(address):
    if os.path.exists(address):
        os.remove(address)


#
import os
import scipy.io as sio

# checking if the directory demo_folder
# exist or not.


if os.path.exists(segments_path + '/' + segments_file_name):
    # Load data
    # load segments from disk

    segments_data = sio.loadmat(segments_path + '/' + segments_file_name)

# print(segments_path)

modelHandler = ModelHandler(configs)

# X_all = segments_data['train_x'].reshape([segments_data['train_x'].shape[0],-1])
train_x = segments_data['train_x']
train_y = segments_data['train_y']  # shape (n_examples,n_classes) - one hot encoding
train_y = train_y.reshape([train_y.shape[0], 1, train_y.shape[1]])  # reshape to (n_examples,1,n_classes)
train_y_all = train_y

test_x = segments_data['test_x']
test_y = segments_data['test_y']
test_y = test_y.reshape([test_y.shape[0], 1, test_y.shape[1]])
test_y_all = test_y

# Now we must concatenate yclass n_grid times along axis 1
for k in range(ngrids - 1):
    train_y_all = np.append(train_y_all, train_y, axis=1)
    test_y_all = np.append(test_y_all, test_y, axis=1)

print(train_y.shape)
print(train_y_all.shape)



num_nan_X = np.count_nonzero(np.isnan(train_x))
print(num_nan_X)

num_nan_Y = np.count_nonzero(np.isnan(train_x))
print(num_nan_Y)


import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling1D, Flatten, MaxPool1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from kymatio.keras import Scattering1D


def choose_model(train_x, train_y, folderPath, complemento=complemento, complemento3=complemento3,
                 complemento2=complemento2):
    from tqdm import tqdm
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score
    from PostProcessing import PostProcessing

    scattering_extract = ModelHandler.loadModel(source_models_folder + 'ST_NILM_' + complemento + '_J' + str(J) + '_Q' + str(Q) + complemento3 + '.h5')  # Load scattering model

    threshold = 0.5
    f1_macro, f1_micro = [], []
    for fold in tqdm(range(1, 11)):
        foldFolderPath = source_models_folder + 'ST_NILM' + '/' + "_J" + str(J) + "_Q" + str(Q) +  complemento2 + '/' + str(fold) + '/'

        train_index = np.load(foldFolderPath + "train_index.npy")
        validation_index = np.load(foldFolderPath + "validation_index.npy")

        bestModel = ModelHandler.loadModel(foldFolderPath + complemento + '.h5', type_weights=None)  # Load model

        #scaler = MaxAbsScaler()
        scaler = MaxAbsScaler()

        scaler.fit(np.squeeze(train_x[train_index], axis=2))
        x_validation = np.expand_dims(scaler.transform(np.squeeze(train_x[validation_index], axis=2)), axis=2)




        x_validation_class = scattering_extract.predict(x_validation)

        # Replacindo infinite values to zero

        x_validation_class = np.nan_to_num(x_validation_class, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalizing


        transformer = StandardScaler().fit(x_validation_class)
        x_validation_class = transformer.transform(x_validation_class)

        final_prediction = []
        final_groundTruth = []
        for xi_nd, yclass in zip(x_validation_class, train_y_all[validation_index]):
            pred = bestModel.predict([np.expand_dims(xi_nd, axis=0)])
            # print("Shape of pred: ", pred.shape)
            prediction = np.max(pred[0],
                                axis=0)  # Withou detection, the first index must be one (Related to classification)
            groundTruth = np.max(yclass, axis=0)

            final_prediction.append(prediction)
            final_groundTruth.append(groundTruth)

            del xi_nd, yclass

        # event_type = np.min(np.argmax(dict_data["y_train"]["type"][validation_index], axis=2), axis=1)

        final_groundTruth = np.array(final_groundTruth)
        final_prediction = np.array(final_prediction)
        final_predictions2 = np.zeros_like(final_prediction)

        final_predictions2[final_prediction > threshold] = 1


        f1_macro.append([f1_score(final_groundTruth, final_predictions2, average='macro', zero_division=0)])

        print(f"Fold {fold}: F1 Macro avg: {np.average(f1_macro[-1]) * 100:.1f}")

    return np.argmax(np.average(f1_macro, axis=1)) + 1, f1_macro

if choose_folder_flag:
    fold, f1_macro = choose_model(train_x, train_y, source_models_folder)

    fold = np.argmax(np.average(f1_macro, axis=1)) + 1
else:
    fold = 1

print(fold)



"""## Evaluates the identification

This step generates a dict with the ground truth and the prediction for each test example
"""

from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


foldFolderPath = source_models_folder + 'ST_NILM' + '/' + "_J" + str(J) + "_Q" + str(Q)  + complemento2 + '/' + str(
    fold) + '/'

train_index = np.load(foldFolderPath + "train_index.npy")
validation_index = np.load(foldFolderPath + "validation_index.npy")

bestModel = ModelHandler.loadModel(foldFolderPath + complemento + '.h5', type_weights=None)  # Load model

scattering_extract = ModelHandler.loadModel(source_models_folder + 'ST_NILM_' + complemento + '_J' + str(J) + '_Q' + str(Q) + complemento3 + '.h5')  # Load scattering model

scaler = StandardScaler()
scaler.fit(np.squeeze(train_x[train_index], axis=2))
x_train = np.expand_dims(scaler.transform(np.squeeze(train_x[train_index], axis=2)), axis=2)
x_validation = np.expand_dims(scaler.transform(np.squeeze(train_x[validation_index], axis=2)), axis=2)
x_test = np.expand_dims(scaler.transform(np.squeeze(test_x, axis=2)), axis=2)

x_train = scattering_extract.predict(x_train)
x_validation = scattering_extract.predict(x_validation)
x_test = scattering_extract.predict(x_test)

# Normalizing
x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
transformer = StandardScaler().fit(x_test)
x_test = transformer.transform(x_test)




final_prediction = []
final_groundTruth = []
for xi_nd, yclass in zip(x_test, test_y):
    # pred = bestModel.predict([np.expand_dims(xi_nd, axis=0)])
    pred = bestModel.predict([np.expand_dims(xi_nd, axis=0)])
    prediction = np.max(pred[0], axis=0)
    groundTruth = np.max(yclass, axis=0)

    final_prediction.append(prediction)
    final_groundTruth.append(groundTruth)

    del xi_nd, yclass

y = {}
y["true"] = final_groundTruth.copy()
y["pred"] = final_prediction.copy()

test_y = np.array(y["true"].copy())
y_pred = np.zeros_like(np.array(y["pred"].copy()))
threshold = 0.5
y_pred[np.array(y["pred"].copy()) > threshold] = 1

print(test_y.shape)



from sklearn.metrics import f1_score

threshold = 0.5
f1_macro = f1_score(np.array(y["true"]) > threshold, np.array(y["pred"]) > threshold, average='macro')
f1_micro = f1_score(np.array(y["true"]) > threshold, np.array(y["pred"]) > threshold, average='micro')

print(f"Fold {fold} - F1 Macro: {f1_macro * 100:.1f}, F1 Micro: {f1_micro * 100:.1f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix


# Calcular a matriz de confusão
confusion_Matrix = confusion_matrix(test_y, y_pred)

print(confusion_Matrix)

import scipy.io as sio

# collect arrays in dictionary
savedict = {
            'Inverter': confusion_Matrix,
            #'class_1': confusion_Matrix[2:4, :],
            #'class_2': confusion_Matrix[4:6, :],
            #'class_3': confusion_Matrix[6:8, :],
            #'class_4': confusion_Matrix[8:10, :]
           }

# save to disk

if not os.path.exists(folderDataPath + "confusion_matrix"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs(folderDataPath + "confusion_matrix")

add2 = folderDataPath + 'confusion_matrix' + "/ST_NILM" +  "_J" + str(J) + "_Q" + str(Q) + complemento + '_' + configs["LOSS"] + '.mat'

remove_if_exists(add2)
# remove_if_exists(root_path + '/trained_models/'+ "confusion_matrix" + "/" + file_name2 + ".mat")

if normalized:
    sio.savemat(folderDataPath + 'confusion_matrix' + "/ST_NILM" +  "_J" + str(J) + "_Q" + str(Q) + complemento + '_' + configs["LOSS"] + '.mat',
                savedict)
else:
    sio.savemat(folderDataPath + 'confusion_matrix' + "/ST_NILM" +  "_J" + str(J) + "_Q" + str(Q) + complemento + '_' + configs["LOSS"] + '.mat', savedict)

# confusion_Matrix.reshape([2,2])

print(confusion_Matrix)
print(complemento)
