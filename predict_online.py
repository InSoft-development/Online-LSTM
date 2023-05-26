import time
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from scipy.special import softmax
from loguru import logger
from utils.fedot import make_forecast
from utils.smooth import exponential_smoothing,double_exponential_smoothing
from utils.utils import load_config, set_gpu, get_len_size
import clickhouse_connect
import scipy

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
    SCALER_LOSS = config['SCALER_LOSS']

    NUM_GROUPS = config['NUM_GROUPS']
    LAG = config['LAG']
    QUERY_DF = config['QUERY_DF']
    CREATE_TABLE = config['CREATE_TABLE']

    ROLLING_MEAN = config['ROLLING_MEAN']
    EXP_SMOOTH = config['EXP_SMOOTH']
    DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']
    ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
    
    POWER_ID = config['POWER_ID']
    POWER_LIMIT = config['POWER_LIMIT']
    
    TRAIN_LEN_FORECAST = config['TRAIN_LEN_FORECAST']
    LEN_FORECAST = config['LEN_FORECAST']
    WINDOW_SIZE = config['WINDOW_SIZE']
    TRESHOLD_ANOMALY = config['TRESHOLD_ANOMALY']
    CONTINUE_COUNT = config['CONTINUE_COUNT']

    client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
    model_list = []
    scaler_list = []
    scaler_loss_list = []
    for i in range(0, NUM_GROUPS):
        model_save = f'{WEIGHTS}/lstm_group_{i}.h5'
        model = load_model(model_save)

        scaler_save = f'{SCALER}/scaler_{i}.pkl'
        scaler = joblib.load(scaler_save)
        scaler_list.append(scaler)
        model_list.append(model)
        scaler_loss_save = f"{SCALER_LOSS}/{config['SCALER_LOSS_NAME']}/scaler_loss{i}.pkl"
        scaler_loss = joblib.load(scaler_loss_save)
        scaler_loss_list.append(scaler_loss)
    prev_df = client.query_df(f"SELECT * FROM lstm_group{i}").tail(1)
    try:
        while True:
            df = client.query_df(QUERY_DF).tail(1)
            logger.info(f'INPUT DATAFRAME: \n {df}')
            logger.info(f'POWER VALUE: {df[POWER_ID][0]}')
            
            if df[POWER_ID][0]>POWER_LIMIT:
                time_df = df['timestamp']
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
                    logger.debug('READ CSV KKS')
                    groups = pd.read_csv(KKS, sep = ';')
                     
                # print(groups['group'])
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
                    # print(group)

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
                    loss_mean = np.mean(np.abs(yhat - preds), axis=1)
                    loss_list += loss.tolist()
                    data = pd.DataFrame(loss, columns=group_list[i].columns)
                    data['timestamp'] = time_df
                    
                    forecast_df = client.query_df(f"SELECT * FROM lstm_group{i}").tail(TRAIN_LEN_FORECAST)
                    # logger.debug(forecast_df)
                    try: 
                        if config['SCALER_LOSS_NAME'] == 'minmax':
                            data['target_value'] = scaler_loss_list[i].transform(loss_mean.reshape(1,-1))
                        elif config['SCALER_LOSS_NAME'] == 'cdf':
                            # logger.debug(loss_mean)
                            scaler_ = scipy.stats.rv_histogram(scaler_loss_list[i])
                            # logger.debug(scaler_loss_list[i])
                            data['target_value'] = scaler_.cdf(loss_mean) * 100
    
                    except Exception as e:
                        logger.error(e)
                        
                    logger.info(f"Target value: {data['target_value']}")
                    # logger.info (loss_mean)
                    
                    predict_val = make_forecast(train_data = np.array(forecast_df['target_value']), len_forecast = LEN_FORECAST, window_size = WINDOW_SIZE )
                    treshold = TRESHOLD_ANOMALY
                    count = 0
                    prob = np.round(data['target_value'],0)
                    continue_count = 0
                    
                    for val in predict_val:
                        count += 1
                        if val > treshold:
                            continue_count += 1
                        else:
                            # count = 0
                            continue_count = 0
                        if continue_count == CONTINUE_COUNT:
                            continue_count = 0
                            break
                    if count<len(predict_val):
                        logger.info
                    print(count)
                    print(treshold)
                    if count == LEN_FORECAST:
                        count = 8640
                    data['prob'] = prob
                    data['count'] = count
                    count = 0
                    # logger.info(prob)
                    # try:
                    #     client.command(f'DROP TABLE lstm_group{i}')  
                    # except Exception as e:
                    #     logger.error(e)
                    if CREATE_TABLE:
                        col_str = '"timestamp"' +' ' +'DateTime, '
                        col_str += '"target_value"' + 'Float64, ' 
                        col_str += '"prob"' + 'Float64, '
                        col_str += '"count"' + 'Float64, '
                        for col in group_list[i].columns:
                            col_str += '"'+ col +'"' + ' ' + 'Float64, '
                        print(col_str)
                        client.command(f'CREATE TABLE lstm_group{i} ({col_str}) ENGINE = Memory')
                    client.insert_df(f"lstm_group{i}", data)
                    prev_df = data
                    logger.info(f'LOSS DATA: \n {data}')
                time.sleep(300)
                # count = 0
                # break
            else:
                client.insert_df(f"lstm_group{i}", prev_df)
                logger.warning(f'POWER_VALUE < {POWER_LIMIT}')
                logger.warning(f'INSERT PREVIOS VALUE')
    except Exception as e:
        logger.error(e)
        # client.disconnect()
        logger.info("disconnected")

if __name__ == "__main__":
    main()