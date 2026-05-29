# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
import tf_keras as keras

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

airlines = pd.read_csv("airlines.csv")
carriers = list(airlines["IATA_CODE"])

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def build_model(train_ds):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(train_ds.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

def string_to_time(time_string):
    if pd.isnull(time_string):
        return np.nan
    try:
        time_string = int(time_string)
        if time_string == 2400:
            time_string = 0
        time_string = "{:04d}".format(time_string)
        return datetime.time(int(time_string[0:2]), int(time_string[2:4]))
    except Exception:
        return np.nan

def func(x):
    return x.hour * 3600 + x.minute * 60 + x.second

def safe_predict(model, input_array):
    try:
        return float(model.predict(input_array, verbose=0).flatten()[0])
    except Exception as e1:
        print(f"[shape (1,6)] {e1}")
    try:
        arr = input_array.reshape(1, 1, input_array.shape[-1])
        return float(model.predict(arr, verbose=0).flatten()[0])
    except Exception as e2:
        print(f"[shape (1,1,6)] {e2}")
    try:
        tensor = tf.constant(input_array, dtype=tf.float32)
        return float(model(tensor, training=False).numpy().flatten()[0])
    except Exception as e3:
        print(f"[model() call] {e3}")
    raise RuntimeError("Không thể predict.")

def do_create_models():
    for carrier in carriers:
        try:
            print("TRAINING:", carrier)
            file_path = f'carriers/carrier{carrier}data.csv'
            if not os.path.exists(file_path):
                print("FILE NOT FOUND:", file_path)
                continue
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df.drop(['Unnamed: 0'], axis=1, inplace=True)
            encoder = LabelEncoder()
            df['ORIGIN_AIRPORT'] = encoder.fit_transform(df['ORIGIN_AIRPORT'])
            train_dataset = df.sample(frac=0.8, random_state=0)
            test_dataset  = df.drop(train_dataset.index)
            train_stats = train_dataset.describe()
            train_stats.pop("ARRIVAL_DELAY")
            train_stats = train_stats.transpose()
            os.makedirs("stats", exist_ok=True)
            train_stats.to_csv(f'stats/train_stats{carrier}.csv')
            train_labels = train_dataset.pop('ARRIVAL_DELAY')
            test_labels  = test_dataset.pop('ARRIVAL_DELAY')
            normed_train = norm(train_dataset, train_stats)
            normed_test  = norm(test_dataset,  train_stats)
            model = build_model(train_dataset)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model.fit(normed_train, train_labels, epochs=100,
                      validation_split=0.2, verbose=0, callbacks=[early_stop])
            loss, mae, mse = model.evaluate(normed_test, test_labels, verbose=0)
            print("MAE:", mae)
            os.makedirs("models", exist_ok=True)
            model.save(f'models/model-{carrier}.h5')
            print("DONE:", carrier)
        except Exception as e:
            print(f"ERROR {carrier}: {e}")

def processInput(input_):
    try:
        carrier = input_["carrier"]
        if not carrier or carrier not in carriers:
            raise ValueError(f"Carrier không hợp lệ: '{carrier}'")

        time_sd = string_to_time(np.int64(input_["sd"]))
        time_sa = string_to_time(np.int64(input_["sa"]))
        if pd.isnull(time_sd) or pd.isnull(time_sa):
            raise ValueError("Thời gian không hợp lệ")
        time_sd = func(time_sd)
        time_sa = func(time_sa)

        csv_path = f'carriers/carrier{carrier}data.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy: {csv_path}")
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)

        encoder = LabelEncoder()
        encoder.fit(df['ORIGIN_AIRPORT'])
        encoded_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        origin = input_["origin"]
        if origin not in encoded_map:
            raise ValueError(f"Sân bay '{origin}' không có trong dữ liệu hãng {carrier}")
        origin_encoded = float(encoded_map[origin])

        stats_path = f'stats/train_stats{carrier}.csv'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Không tìm thấy: {stats_path}")
        train_stats = pd.read_csv(stats_path, index_col=0)

        row = {
            "time_insec_dep": float(time_sd),
            "time_insec_arr": float(time_sa),
            "ORIGIN_AIRPORT": origin_encoded,
            "DEPARTURE_DELAY": float(input_["ddelay"]),
            "DISTANCE":        float(input_["dist"]),
            "weekday":         float(input_["dayOfWeek"]),
        }
        df_input  = pd.DataFrame([row])
        df_normed = norm(df_input, train_stats).fillna(0)
        input_array = df_normed.values.astype(np.float32)

        model_path = f'models/model-{carrier}.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy: {model_path}")

        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ MODEL LOADED: {carrier}")

        return input_array, model

    except Exception as e:
        print(f"❌ PROCESS INPUT ERROR [{input_.get('carrier','?')}]: {e}")
        return None, None

def predict_delay(input_):
    try:
        data, model = processInput(input_)
        if data is None or model is None:
            return None
        return round(safe_predict(model, data), 2)
    except Exception as e:
        print(f"❌ PREDICTION ERROR: {e}")
        return None