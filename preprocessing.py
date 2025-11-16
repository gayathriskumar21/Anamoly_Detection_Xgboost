import pandas as pd
import numpy as np


def time_to_cyclic(time_str):
    hh, mm = map(int, time_str.split(':'))
    minutes = hh * 60 + mm
    sin_val = np.sin(2 * np.pi * minutes / (24 * 60))
    cos_val = np.cos(2 * np.pi * minutes / (24 * 60))
    return pd.Series([sin_val, cos_val])


def preprocess_train(train_csv, encoding='latin1'):
    df = pd.read_csv(train_csv, encoding=encoding)
    df[['Time_sin', 'Time_cos']] = df['Time'].apply(time_to_cyclic)

    # Example: One-hot encode 'CellName'
    cell_encoded = pd.get_dummies(df['CellName'], prefix='CellName')
    df = pd.concat([df, cell_encoded], axis=1)

    # Drop unused columns for training features
    df = df.drop(columns=['Time', 'CellName'])

    # Normalize or scale - example using min-max for demonstration
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(df)

    # Returning encoders/scalers as None if not applicable or add your own
    return X_train_norm, None, None, scaler, None


def preprocess_uploaded_data(df, scaler, encoder):
    df[['Time_sin', 'Time_cos']] = df['Time'].apply(time_to_cyclic)
    cell_encoded = encoder.transform(df[['CellName']])
    cell_df = pd.DataFrame(cell_encoded, columns=encoder.get_feature_names_out(['CellName']))
    df = pd.concat([df, cell_df], axis=1)
    df = df.drop(columns=['Time', 'CellName', 'Hour'], errors='ignore')
    df = df.ffill().bfill()
    X_scaled = scaler.transform(df)
    return X_scaled
