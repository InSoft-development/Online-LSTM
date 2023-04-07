import time
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from utils.smooth import exponential_smoothing,double_exponential_smoothing
from utils.utils import load_config, set_gpu, get_len_size
import clickhouse_connect

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--csv_kks', type=bool, default=False)
    opt = parser.parse_args()
    config = load_config(f'{opt.config_path}/config_online.yaml')

    KKS = config['KKS']
    WEIGHTS = config['WEIGHTS']
    SCALER = config['SCALER']

    NUM_GROUPS = config['NUM_GROUPS']
    LAG = config['LAG']
    QUERY_DF = config['QUERY_DF']

    ROLLING_MEAN = config['ROLLING_MEAN']
    EXP_SMOOTH = config['EXP_SMOOTH']
    DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']
    ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
    
    POWER_ID = config['POWER_ID']
    POWER_LIMIT = config['POWER_LIMIT']

    client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
    model_list = []
    scaler_list = []
    for i in range(0, NUM_GROUPS):
        model_save = f'{WEIGHTS}/lstm_group_{i}.h5'
        model = load_model(model_save)

        scaler_save = f'{SCALER}/scaler_{i}.pkl'
        scaler = joblib.load(scaler_save)
        scaler_list.append(scaler)
        model_list.append(model)
    prev_df = client.query_df(f"SELECT * FROM lstm_group{i}").tail(1)
    try:
        while True:
            df = client.query_df(QUERY_DF).tail(1)
            print(df[POWER_ID][0])
            if df[POWER_ID][0]>POWER_LIMIT:
                df = df.iloc[:, :-1]
                if ROLLING_MEAN:
                    rolling_mean = df.rolling(window=ROLLING_MEAN_WINDOW).mean()
                if EXP_SMOOTH:
                    for i in df.columns:
                        df[str(i)] = exponential_smoothing(df[str(i)].to_numpy(), alpha=0.2)
                if DOUBLE_EXP_SMOOTH:
                    for i in df.columns:
                        df[str(i)] = double_exponential_smoothing(df[str(i)].to_numpy(), alpha=0.02, beta=0.09)
                if not opt.csv_kks:
                    groups = client.query_df("SELECT * FROM kks")
                else:
                    groups = pd.read_csv(KKS, sep = ';')
                     
                print(groups['group'])
                group_list = []
                sum = 0
                for i in range(0, NUM_GROUPS):
                    group = groups[groups['group'] == i]
                    if i != 0:
                        group = group.append(groups[groups['group'] == 0])
                    sum += len(group)
                    if len(group) == 0:
                        continue
                    group = df[group['kks']]
                    scaler = scaler_list[i]
                    scaled_data = pd.DataFrame(
                        data=scaler.transform(group),
                        columns=group.columns)
                    group_list.append(scaled_data)
                    print(group)

                loss_list = []
                for i in range(0, len(group_list)):
                    X = group_list[i].to_numpy()
                    model = model_list[i]
                    len_size = get_len_size(LAG, X.shape[0])
                    X = X[:len_size].reshape(int(X.shape[0] / LAG), int(LAG), X.shape[1])
                    preds = model.predict(X, verbose=1)
                    preds = preds[:, 0, :]
                    yhat = X[:, 0, :]
                    loss = abs(yhat - preds)
                    loss_list += loss.tolist()
                    data = pd.DataFrame(loss, columns=group_list[i].columns)
                    client.insert_df(f"lstm_group{i}", data)
                    prev_df = data
                    print(data)
                time.sleep(5)
            else:
                client.insert_df(f"lstm_group{i}", prev_df)
    finally:
        client.disconnect()
        print("disconnected")

if __name__ == "__main__":
    main()