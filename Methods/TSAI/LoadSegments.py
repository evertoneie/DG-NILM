class LoadSegments:
    def __init__(self, configs):
        try:
            self.m_ngrids = configs["N_GRIDS"]
            self.m_nclass = configs["N_CLASS"]
            self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
            self.m_marginRatio = configs["MARGIN_RATIO"]
            self.m_gridLength = int(self.m_signalBaseLength / self.m_ngrids)
            self.configs = configs

            if "USE_NO_LOAD" in self.configs and self.configs["USE_NO_LOAD"] == True:
                self.m_nclass += 1
        except:
            print("Erro no dicionÃ¡rio de configuraÃ§Ãµes")
            exit(-1)


    def load_segments(self, normalized, subsets_selector):


        source_models_folder = './ST-NILM/LIDE/trained_models/'

        J = 4
        Q = 8

        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.model_selection import train_test_split, KFold
        from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
        from tensorflow.keras.optimizers import Adam
        import numpy as np
        import os
        import pickle
        import sys

        sys.path.append("ST-NILM/src")
        from DataHandler import DataHandler
        from ModelHandler import ModelHandler
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        from skmultilearn.model_selection import iterative_train_test_split
        from sklearn.model_selection import KFold
        from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


        ngrids = self.configs["N_GRIDS"]
        signalBaseLength = self.configs["SIGNAL_BASE_LENGTH"]
        trainSize = self.configs["TRAIN_SIZE"]
        folderDataPath = self.configs["FOLDER_DATA_PATH"]
        folderPath = self.configs["FOLDER_PATH"]

        dataHandler = DataHandler(self.configs)

        """#### Load Data"""

        root_path = "./ST-NILM"
        data_path = "./segments"
        segments_path = "./segments"

        # normalized = 0
        # subsets_selector = 2
        win_size = 512
        # subsets_selector
        # 0 - Aggregated
        # 1 - Individual
        # 2 - All

        sys.path.append(root_path)

        #normalized = 1
        #subsets_selector = 2

        # subsets_selector
        # 0 - Aggregated
        # 1 - Individual
        # 2 - All
        reduced_dataset_flag = False  # Essa flag é para o caso em que fique muito grande o conjunto de treino e teste
        load_existing_segments = False

        if subsets_selector == 0:
            complemento = "Detect_Multiclass_Inverter_aggregated_512"
            complemento2 = "_aggregated"
        elif subsets_selector == 1:
            complemento = "Detect_Multiclass_Inverter_individual_512"
            complemento2 = "_individual"
        elif subsets_selector == 2:
            complemento = "Detect_Multiclass_Inverter_all_512"
            complemento2 = "_all"

        if reduced_dataset_flag:
            complemento = "reduced_" + complemento

        if normalized:
            complemento = "normalized_" + complemento


        complemento3 = ""
        complemento2 = complemento2 + '_' + self.configs["LOSS"]
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

        modelHandler = ModelHandler(self.configs)


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

        #configs["PERCENTUAL"][0]

        return train_x, train_y, test_x, test_y
