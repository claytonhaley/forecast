from pandas_datareader import data as pdr
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import SGD


# ---- PREPROCESS ----
def _preprocess(ticker, start, end):
    """
    Loads the ticker data, selects columns for training, and scales the data appropriately.
    
            Parameters:
                    ticker   (string): stock ticker
                    start    (string): string formatted date
                    end      (string): string formatted date
            Returns:
                    scaler   (scaler obj): scaler obj used later
                    df       (pd.DataFrame): dataframe without Date column 
                    dates    (pd.Series): dates from dataframe
                    final_df (pd.DataFrame): dataframe used for training
            Except:
                    st.error: Ticker not available message
    """

    try:
        df = pdr.get_data_yahoo(f"{ticker}", start=f"{start}", end=f"{end}")
    except:
        st.error("Ticker Not Available")
    
    df = df[['High', 'Low', 'Open', 'Volume', 'Close']]
    df.reset_index(inplace=True)

    dates = df['Date']
    df = df.drop('Date', axis=1)

    # Scale the data and create training dataset
    scaler = StandardScaler()
    final_df = scaler.fit_transform(df)
    return scaler, df, dates, final_df


# ---- CREATE TRAINING DATASET ----
def _create_dataset(dataset, future_days, past_days):
    """
    Creates training datasets based on the number of lookback days and prediction days.
    
            Parameters:
                    dataset      (np.ndarray): stock ticker
                    future_days  (int): string formatted date
                    past_days    (int): string formatted date
            Returns:
                    X_train      (np.ndarray): input values
                    y_train      (np.ndarray): target values 
    """

    X_train, y_train = [], []
    for i in range(past_days, len(dataset) - future_days + 1):
        X_train.append(dataset[i - past_days:i, 0:dataset.shape[1] - 1])
        y_train.append(dataset[i + future_days - 1:i + future_days, 4])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train


# ---- BUILD LSTM NEURAL NETWORK ----
def _build_model(data, lookback_days, n_epochs, X_train, y_train):
    """
    Creates training datasets based on the number of lookback days and prediction days.
    
            Parameters:
                    data            (np.ndarray): training data
                    lookback_days   (int): days considered for training
                    n_epochs        (int): training epochs
                    X_train         (np.ndarray): input values
                    y_train         (np.ndarray): target values
            Returns:
                    model           (model obj): trained model
                    history:        (history obj): model history/details
    """

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(lookback_days, data.shape[1]-1), return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0), loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    
    history = model.fit(X_train, y_train, epochs=n_epochs, callbacks=early_stop,
                            batch_size=16, validation_split=0.1, verbose=0)

    return model, history


# ---- GENERATE PREDICTIONS ----
def _generate_predictions(data, model, scaler, dates, 
                            n_future, n_back, X_train, y_train):
    """
    Creates training datasets based on the number of lookback days and prediction days.
    
            Parameters:
                    data                       (np.ndarray): training data
                    model                      (model obj): trained model
                    scaler                     (scaler obj): scaler used to reverse transformed values
                    dates                      (np.ndarray): input values
                    n_future                   (int): number of future days predicted
                    n_back                     (int): number of considered past days
                    X_train                    (np.ndarray): input values
                    y_train                    (np.ndarray): target values
            Returns:
                    mse                        (float): root mean squared error
                    PREDICTIONS_FUTURE         (pd.DataFrame): future predictions
                    PREDICTION_TRAIN           (pd.DataFrame): training predictions
    """                    

    # Generate all y_train values
    y_train_copies = np.repeat(y_train, data.shape[1], axis=-1)
    y_train_actual = scaler.inverse_transform(y_train_copies)[:,0]
    
    # Generate all y_train predictions
    all_predictions_train = model.predict(X_train)
    all_y_train_pred_copies = np.repeat(all_predictions_train, data.shape[1], axis=-1)
    all_y_train_pred_actual = scaler.inverse_transform(all_y_train_pred_copies)[:,0]

    # Generate future and previous predictions based on lookback and future days
    predictions_future = model.predict(X_train[-n_future:])
    predictions_train = model.predict(X_train[n_back:])

    # Inverse transform predictions_train
    train_predict_copies = np.repeat(predictions_train, data.shape[1], axis=-1)
    y_pred_train = scaler.inverse_transform(train_predict_copies)[:,0]

    # Inverse transform predictions_future
    train_future_copies = np.repeat(predictions_future, data.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(train_future_copies)[:,0]

    # Dates for future predictions
    datelist_future = pd.date_range(list(dates)[-1], periods=n_future, freq='1d').tolist()

    # Final data frames for plotting
    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Close']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Close']).set_index(pd.Series(dates[2 * n_back + n_future - 1:]))

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_train_actual, all_y_train_pred_actual))

    return rmse, PREDICTIONS_FUTURE, PREDICTION_TRAIN


def _pred_plot(df, dates, start_date_plot, future_preds, train_preds):
    """
    Creates training datasets based on the number of lookback days and prediction days.
    
            Parameters:
                    df                       (pd.DataFrame): original dataframe
                    dates                    (pd.Series): dates from dataframe
                    start_date_plot          (string): date to start plot
                    future_preds             (pd.DataFrame): future predictions
                    train_preds              (pd.DataFrame): training predictions
            Returns:
                    st.pyplot(fig)
    """

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


# -------- MAIN FUNCTION --------
def main():
    st.set_page_config(page_title="Stock Market Prediction",
                    page_icon=":chart_with_upwards_trend:",
                    layout="wide")

    st.title(":chart_with_upwards_trend: Stock Market Prediction with LSTMs")
    st.markdown("##")
    left_column, right_column = st.columns((1, 2))

    # ---- SIDEBAR ---- 
    st.sidebar.title("Build your own ML Model for Time Series Prediction")
    ticker_file = open("tickers.txt", "r")
    ticker_options = ticker_file.read().splitlines()
    ticker_file.close()

    with st.sidebar.form(key='Params'):
        # Param options
        ticker_name = st.sidebar.selectbox("Select Ticker", ticker_options)

        start_date = str(st.sidebar.date_input("Chose a start date for data collection"))
        end_date = str(st.sidebar.date_input("Chose an end date for data collection"))

        lookback_days = st.sidebar.slider("Past days considered for training", min_value=1, max_value=1000)
        lookahead_days = st.sidebar.slider("Days ahead to predict", min_value=1, max_value=100)

        epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100)

        submit = st.form_submit_button('Run Model')

        if submit:
            try:
                # ---- DATA PREPROCESSING ----
                scaler, original_df, all_dates, final_df = _preprocess(ticker_name, start_date, end_date)

                X_train, y_train = _create_dataset(final_df, lookahead_days, lookback_days)
            
                # Instantiate Model Architecture
                with st.spinner('Training Model...'):
                    model, history = _build_model(final_df, lookback_days, epochs, X_train, y_train)
                
                with left_column:
                    fig = plt.figure()
                    plt.plot(history.history['loss'], label='Training loss')
                    plt.plot(history.history['val_loss'], label='Validation loss')
                    plt.legend()
                    st.pyplot(fig)

                # Generating Predictions
                with st.spinner('Generating Predictions...'):
                    rmse, future_predictions, train_predictions = _generate_predictions(original_df, model, scaler, all_dates, 
                                                                lookahead_days, lookback_days, X_train, y_train)

                    with right_column:
                        _pred_plot(original_df, all_dates, start_date, future_predictions, train_predictions)

                st.success('Done!')

                # Generating RMSE
                with left_column:
                    st.warning(f"Stopped at {len(history.history['loss'])} epochs (loss failed to improve)")
                    st.info(f"RMSE: {rmse}")
                
            except:
                st.error("Please choose the appropriate parameters")


if __name__ == "__main__":
    main()