from pandas_datareader import data as pdr
import datetime
import math
from pylab import rcParams
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def _create_dataset(dataset, future_days, past_days):
    X_train, y_train = [], []
    for i in range(past_days, len(dataset) - future_days + 1):
        X_train.append(dataset[i - past_days:i, 0:dataset.shape[1] - 1])
        y_train.append(dataset[i + future_days - 1:i + future_days, 4])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train


def _build_model(data, lookback_days, X_train, y_train):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(lookback_days, data.shape[1]-1), return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
    
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    history = model.fit(X_train, y_train, epochs=30, 
                            batch_size=16, validation_split=0.1, verbose=0)

    return model, history

def _preprocess(ticker, start, end):
    df = pdr.get_data_yahoo(f"{ticker}", start=f"{start}", end=f"{end}")
    df = df[['High', 'Low', 'Open', 'Volume', 'Close']]
    df.reset_index(inplace=True)

    dates = df['Date']
    df = df.drop('Date', axis=1)

    # Scale the data and create training dataset
    scaler = StandardScaler()
    final_df = scaler.fit_transform(df)

    return scaler, df, dates, final_df

def _generate_predictions(data, model, scaler, dates, 
                            n_future, n_back, X_train, y_train):
    y_train_copies = np.repeat(y_train, data.shape[1], axis=-1)
    y_train_actual = scaler.inverse_transform(y_train_copies)[:,0]
    
    all_predictions_train = model.predict(X_train)
    all_y_train_pred_copies = np.repeat(all_predictions_train, data.shape[1], axis=-1)
    all_y_train_pred_actual = scaler.inverse_transform(all_y_train_pred_copies)[:,0]

    predictions_future = model.predict(X_train[-n_future:])
    predictions_train = model.predict(X_train[n_back:])

    train_predict_copies = np.repeat(predictions_train, data.shape[1], axis=-1)
    y_pred_train = scaler.inverse_transform(train_predict_copies)[:,0]

    train_future_copies = np.repeat(predictions_future, data.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(train_future_copies)[:,0]

    datelist_future = pd.date_range(list(dates)[-1], periods=n_future, freq='1d').tolist()

    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Close']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Close']).set_index(pd.Series(dates[2 * n_back + n_future - 1:]))

    mse = math.sqrt(mean_squared_error(y_train_actual, all_y_train_pred_actual))

    return mse, PREDICTIONS_FUTURE, PREDICTION_TRAIN


def _pred_plot(df, dates, start_date_plot, future_preds, train_preds):
    dates = list(dates)
    df['Date'] = dates
    df.set_index('Date', inplace=True)
    
    fig = plt.figure()
    plt.plot(future_preds.index, future_preds['Close'], color='r', label='Predicted Stock Price')
    plt.plot(train_preds.loc[start_date_plot:].index, train_preds.loc[start_date_plot:]['Close'], color='orange', label='Training predictions')
    plt.plot(df.loc[start_date_plot:].index, df.loc[start_date_plot:]['Close'], color='b', label='Actual Stock Price')

    plt.axvline(x = min(future_preds.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predcitions and Actual Stock Prices', family='Arial', fontsize=18)
    plt.xlabel('Timeline', family='Arial', fontsize=14)
    plt.ylabel('Stock Price Value', family='Arial', fontsize=14)
    plt.xticks(rotation=45, fontsize=8)
    st.pyplot(fig)


## -------- MAIN FUNCTION --------
def main():

    st.set_page_config(page_title="Stock Market Prediction",
                    page_icon=":chart_with_upwards_trend:",
                    layout="wide")

    st.title(":chart_with_upwards_trend: Stock Market Prediction with LSTMs")
    st.markdown("##")
    left_column, right_column = st.columns(2)

    # ---- SIDEBAR ---- 
    st.sidebar.title("Build your own ML Model for Time Series Prediction")
    ticker_file = open("tickers.txt", "r")
    ticker_options = ticker_file.read().splitlines()
    ticker_file.close()

    with st.sidebar.form(key='Params'):
        ticker_name = st.sidebar.selectbox("Select Ticker", ticker_options)

        start_date = str(st.sidebar.date_input("Chose a start date for data collection"))
        end_date = str(st.sidebar.date_input("Chose an end date for data collection"))

        lookback_days = st.sidebar.slider("Past days considered for training", min_value=1, max_value=1000)
        lookahead_days = st.sidebar.slider("Days ahead to predict", min_value=1, max_value=100)

        start_date_plot = str(st.sidebar.date_input("Start Date for Prediction Plot"))

        submit = st.form_submit_button('Run Model')

        if submit:
            # try:
            # ---- DATA PREPROCESSING ----
            scaler, original_df, all_dates, final_df = _preprocess(ticker_name, start_date, end_date)

            X_train, y_train = _create_dataset(final_df, lookahead_days, lookback_days)

            # Instantiate Model Architecture
            with st.spinner('Training Model...'):
                model, history = _build_model(final_df, lookback_days, X_train, y_train)
            
            with left_column:
                fig = plt.figure()
                plt.plot(history.history['loss'], label='Training loss')
                plt.plot(history.history['val_loss'], label='Validation loss')
                plt.legend()
                st.pyplot(fig)

            # Generating Predictions
            with st.spinner('Generating Predictions...'):
                mse, future_predictions, train_predictions = _generate_predictions(original_df, model, scaler, all_dates, 
                                                            lookahead_days, lookback_days, X_train, y_train)

                with right_column:
                    _pred_plot(original_df, all_dates, start_date_plot, future_predictions, train_predictions)

            st.success('Done!')

            # Generating RMSE
            with left_column:
                st.subheader("Root Mean Square Error:")
                st.subheader(f"RMSE {mse}")
                
            # except:
            #     st.error("Please choose the appropriate parameters")

if __name__ == "__main__":
    main()