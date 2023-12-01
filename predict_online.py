import warnings
warnings.filterwarnings('ignore')



import time
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
import scipy
import argparse
import json
from loguru import logger
from tensorflow.keras.models import load_model
from tqdm import tqdm
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import clickhouse_connect
from utils.smooth import exponential_smoothing, double_exponential_smoothing
from utils.utils import load_config, get_len_size, hist_threshold, get_anomaly_interval
import sys
import matplotlib.pyplot as plt

VERSION = "1.0.1"

os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
        
def set_tf_config():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Use argparse to get command line arguments
def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--file_format', type=str, default='db')
    parser.add_argument('--csv_kks', type=bool, default=False)
    parser.add_argument("-v", "--version", action="store_true", help="вывести версию программы")

    return parser.parse_args()

def ensure_directories_exist(config):
    os.makedirs(config['CSV_SAVE_RESULT_LOSS'], exist_ok=True)
    os.makedirs(config['CSV_SAVE_RESULT_PREDICT'], exist_ok=True)
    os.makedirs(config['CSV_DATA'], exist_ok=True)
    os.makedirs(config['JSON_DATA'], exist_ok=True)
    os.makedirs(config['SCALER_LOSS'], exist_ok=True)

def load_dataset(file_format, file):
    if file_format == 'db':
        client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
        return client.query_df(file)
    elif file_format == 'csv':
        return pd.read_csv(file)
    elif file_format == 'sqlite':
        cnx = sqlite3.connect(file)
        return pd.read_sql_query("SELECT * FROM 'data_train'", cnx)

def load_models_and_scalers(num_groups, weights_path, scaler_path):
    model_list = []
    scaler_list = []
    for i in range(num_groups):
        model_file = f'{weights_path}/lstm_group_{i}.h5'
        model = load_model(model_file)

        scaler_file = f'{scaler_path}/scaler_{i}.pkl'
        scaler = joblib.load(scaler_file)
        scaler_list.append(scaler)
        model_list.append(model)
    
    return model_list, scaler_list

def load_groups(opt, kks):
    if not opt.csv_kks:
        client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
        groups = client.query_df("SELECT * FROM kks")
    else:
        groups = pd.read_csv(kks, sep=';')
    return groups
        
def preprocess_data(df, power_id, power_limit, config):
    time_ = df['timestamp']
    df = df[df[power_id] > power_limit]
    test_time = df['timestamp']
    df.drop(columns=['timestamp'], inplace=True)

    if config['MEAN_NAN']:
        df.fillna(df.mean(), inplace=True)

    if config['DROP_NAN']:
        df.dropna(inplace=True)

    if config['ROLLING_MEAN']:
        df.rolling(window=config['ROLLING_MEAN_WINDOW']).mean()

    if config['EXP_SMOOTH']:
        df = df.apply(exponential_smoothing, alpha=0.2)

    if config['DOUBLE_EXP_SMOOTH']:
        df = df.apply(double_exponential_smoothing, alpha=0.02, beta=0.09)

    return df, test_time, time_

def main():
# Load configuration
    set_tf_config()
    opt = get_options()
    if opt.version:
        print("Версия predict offline:", VERSION)
        sys.exit()
    config = load_config(f'{opt.config_path}/config_offline.yaml')
    ensure_directories_exist(config)
    test_df = load_dataset(opt.file_format, config['TEST_FILE'])
    logger.info(f"DATAFRAME: \n {test_df}")
    model_list, scaler_list = load_models_and_scalers(config['NUM_GROUPS'], config['WEIGHTS'], config['SCALER'])
    groups = load_groups(opt, config['KKS'])
    logger.info(groups)
    test_df, test_time, time_ = preprocess_data(test_df, config['POWER_ID'], config['POWER_LIMIT'], config)


    g = groups[groups['group']==0]
    zero_group = g['kks']
    group_list = []
    unscaled = []
    
    for i in range(config['NUM_GROUPS']):
        group = groups[groups['group'] == i]
        names = groups['name'][groups['group'] == i]

        if i != 0:
            group = group.append(groups[groups['group'] == 0])

        if len(group) == 0:
            continue

        group = test_df[group['kks']]
        scaler = scaler_list[i]
        scaled_data = pd.DataFrame(
            data=scaler.transform(group),
            columns=group.columns
        )

        group_list.append(scaled_data)
        unscaled.append(group)

    if config['AUTO_DROP_LIST']:
        DROP_LIST = list(group_list[0].columns)
        logger.debug(f"DROP_LIST: \n {DROP_LIST}")

    dict_list = []  # Create an empty list to store dictionaries
    for i, (data, scaled_data, model) in enumerate(zip(unscaled, group_list, model_list)):
        logger.info(f'GROUP {i}')
        X = scaled_data.to_numpy()
        len_size = get_len_size(config['LAG'], X.shape[0])
        X = X[:len_size].reshape(int(X.shape[0] / config['LAG']), config['LAG'], X.shape[1])
        preds = model.predict(X, verbose=1)
        preds = preds[:, 0, :]
        yhat = X[:, 0, :]
        loss = np.mean(np.abs(yhat - preds), axis=1)
        each_loss = np.abs(yhat - preds)
        df_lstm = pd.DataFrame(each_loss, columns=scaled_data.columns)
        
        logger.info(f"Scaling loss {config['SCALER_LOSS_NAME']}")
        
        def scaler_loss(target_value, scaler_name, range_loss = 100):
            if scaler_name == 'cdf':
                hist = np.histogram(target_value, bins=range_loss)
                # logger.debug(target_value)
                scaler_loss = scipy.stats.rv_histogram(hist) 
                # logger.debug(hist)
                target_value = scaler_loss.cdf(target_value)*range_loss
                scaler_loss = hist
            elif scaler_name == 'minmax':
                scaler_loss = MinMaxScaler(feature_range=(0, range_loss))
                loss_2d = np.reshape(target_value, (-1,1))
                scaler_loss.fit(loss_2d)
                target_value = scaler_loss.transform(loss_2d)
            return target_value, scaler_loss
        
        target_value, scaler_loss = scaler_loss(loss, config['SCALER_LOSS_NAME'])
        df_lstm['target_value'] = target_value   
        joblib.dump(scaler_loss, f"{config['SCALER_LOSS']}/scaler_loss{i}.pkl")

        df_lstm.index = test_time[:len_size][::config['LAG']]
        df_timestamps = pd.DataFrame()
        df_timestamps['timestamp'] = time_
        df_lstm = pd.merge(df_lstm, df_timestamps, on='timestamp', how='right')
        
        # if config['POWER_FILL'] == 'last_value':
        #         df_lstm.fillna(method='ffill', inplace=True)  # Заполняем пропущенные значения последними непустыми значениями
        if config['POWER_FILL']:
            
            df_lstm.fillna(0, inplace=True)    # Можно указать 'zeroes', чтобы заполнять нулями
            # limit = 240
            print()
            def fill_by_power(df, limit, count_next,power_id):
                df['POWER'] = data[config['POWER_ID']]
                for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                    # print(row[power_id])
                    # print(row['POWER'])
                    if row['POWER'] < limit:
                        # print(row[power_id])
                        # if row['target_value'] is None:
                        #     pass
                        next_rows = df.loc[index: index + count_next - 1]
                        # print((next_rows[power_id] > limit).any())
                        if (next_rows['POWER'] > limit).any():
                            print('!')
                            if index != 0:
                                # print(df.at[index - 1, 'target_value'])
                                df.loc[index, 'target_value'] = df.loc[index - 1, 'target_value']
                            else:
                                df.loc[index, 'target_value'] = 0
                        else:
                            # print('!')
                            df.loc[index, 'target_value'] = 0
                    else:
                        # print(row['target_value'])
                        df.loc[index] = row
                return df
            
            fill_by_power(df_lstm,config['POWER_LIMIT'],config['COUNT_NEXT'],config['POWER_ID'])
            
            # df_lstm.to_csv('test.csv')
            
        try:
            if i != 0:
                df_lstm = df_lstm.drop(columns=config['DROP_LIST'])

        except:
            logger.info('No columns to drop')

        try:
            df_lstm = df_lstm.drop(columns=config['DROP_LIST'])
            logger.info(f'Drop {DROP_LIST}')

        except:
            logger.info('No columns to drop')
        df_lstm = df_lstm.replace(0, np.nan)
        
        df_loss = pd.DataFrame()
        df_loss['target_value'] = df_lstm['target_value']
        df_loss['timestamp'] = df_lstm['timestamp']
    
        data.to_csv(f"{config['CSV_DATA']}/group_{i}.csv")
        df_lstm = df_lstm.drop(columns=zero_group)
        df_lstm = df_lstm.drop(columns=['target_value','POWER'])
        # df_lstm = df.lstm.drop(columns)
        # df_lstm.drop(columns = )
        df_lstm.to_csv(f"{config['CSV_SAVE_RESULT_LOSS']}/loss_{i}.csv", index = False)
        logger.info(f"LOSS DATA FRAME: \n {df_lstm}")
        
        df_loss.to_csv(f"{config['CSV_SAVE_RESULT_PREDICT']}/predict_{i}.csv", index = False)
        logger.info(f"PREDICT DATA'FRAME: \n {df_loss}")
        
        dict_list = []  # Reset the dict_list for the next group
        
main()
