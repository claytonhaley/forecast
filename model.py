import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam


class ModelIntegration(HyperModel):

    def __init__(self, ticker_name, start_date, end_date, 
                    lookback_days, lookahead_days, epochs):
        """

        """
        self.ticker_name = ticker_name
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.lookahead_days = lookahead_days
        self.epochs = epochs

    # ---- PREPROCESS ----
    def _preprocess(self):
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
            yahoo_financials = YahooFinancials(f"{self.ticker_name}")           
        except:
            st.error("Ticker Not Available")
        
        data = yahoo_financials.get_historical_price_data(f"{self.start_date}", f"{self.end_date}", "daily")
        full_data = pd.DataFrame(data[f"{self.ticker_name}"]['prices'])

        self.dates = full_data['formatted_date']
        self.df = full_data[['high', 'low', 'open', 'volume', 'adjclose', 'close']]
        
        return self.df, self.dates


    # ---- CREATE TRAINING DATASET ----
    def _create_dataset(self):
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
        dataset = self.df.to_numpy()
        self.X_scaler = MinMaxScaler(feature_range=(-1,1))
        X_scaled = self.X_scaler.fit_transform(dataset[:,0:dataset.shape[1]-1])
        
        self.y_scaler = MinMaxScaler(feature_range=(-1,1))
        y_scaled = self.y_scaler.fit_transform(dataset[:,-1].reshape(-1, 1))

        dataset = np.hstack((X_scaled, y_scaled))

        X_train, y_train = [], []
        for i in range(self.lookback_days, len(dataset) - self.lookahead_days + 1):
                X_train.append(dataset[i - self.lookback_days:i, 0:dataset.shape[1] - 1])
                y_train.append(dataset[i + self.lookahead_days - 1:i + self.lookahead_days, 5])

        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        return self.X_train, self.y_train, self.X_scaler, self.y_scaler


    # ---- BUILD LSTM NEURAL NETWORK ----
    def _build_default_model(self):
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
        model.add(Conv1D(32, self.lookback_days-9, activation='relu', input_shape=(self.lookback_days, self.X_train.shape[2])))
        model.add(MaxPooling1D(1,1))
        model.add((LSTM(64, return_sequences=True)))
        model.add((LSTM(32)))
        model.add(Dense(16))
        model.add(Dense(1))

        model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

        history = model.fit(self.X_train, self.y_train, epochs=self.epochs, callbacks=[early_stop], validation_split=0.1, verbose=0)

        return model, history


    def _build_tuned_model(self, hp):
        """

        """
        
        model = Sequential()
        model.add(Conv1D(hp.Int(name='input_units_1', min_value=16, max_value=128, step=16), 
                        hp.Int(name='input_units_2', min_value=16, max_value=self.lookback_days-9, step=16), activation='relu', input_shape=(self.lookback_days, self.X_train.shape[2])))
        model.add(MaxPooling1D(1,1))
        model.add((LSTM(hp.Int(name='lstm_1', min_value=16, max_value=128, step=16), return_sequences=True)))
        model.add((LSTM(hp.Int(name='lstm_2', min_value=16, max_value=128, step=16))))
        model.add(Dense(hp.Int(name='dense_1', min_value=16, max_value=128, step=16)))
        model.add(Dense(1))
        
        model.compile(optimizer = Adam(learning_rate=hp.Float(name='learning_rate', min_value=0.001, max_value=0.1, step=0.001)), 
                        loss='mean_squared_error', 
                        run_eagerly=True)
        

        return model

    def _run_tuned_model(self):
        """

        """

        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        
        tuner = RandomSearch(
                self._build_tuned_model,
                objective='val_loss',
                max_trials=3,
                executions_per_trial=3,
                )
                
        tuner.search(self.X_train, self.y_train, epochs=5, validation_split=0.1, callbacks=[early_stop])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(self.X_train, self.y_train, epochs=100, validation_split=0.1, callbacks=[early_stop])

        return model, history

    # ---- GENERATE PREDICTIONS ----
    def _generate_predictions(self, model):
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
        all_predictions_train = model.predict(self.X_train)
        all_y_train_pred_actual = self.y_scaler.inverse_transform(all_predictions_train)


        predictions_train = model.predict(self.X_train[self.lookback_days:])

        y_pred_train = self.y_scaler.inverse_transform(predictions_train)


        predictions_future = model.predict(self.X_train[-self.lookahead_days:])
        y_pred_future = self.y_scaler.inverse_transform(predictions_future)

        # Dates for future predictions
        datelist_future = pd.date_range(list(self.dates)[-1], periods=self.lookahead_days, freq='1d').tolist()

        # Final data frames for plotting
        self.PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['close']).set_index(pd.Series(datelist_future))
        self.PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['close']).set_index(pd.Series(self.dates[2 * self.lookback_days + self.lookahead_days - 1:]))
        self.PREDICTION_TRAIN.index = pd.DatetimeIndex(self.PREDICTION_TRAIN.index)

        # Calculate RMSE and R^2
        rmse = math.sqrt(mean_squared_error(self.y_scaler.inverse_transform(self.y_train), all_y_train_pred_actual))
        r2 = r2_score(self.y_scaler.inverse_transform(self.y_train), all_y_train_pred_actual)

        return rmse, r2, self.PREDICTIONS_FUTURE, self.PREDICTION_TRAIN


    def _pred_plot(self):
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

        # Plot parameters
        START_DATE_FOR_PLOTTING = self.start_date

        dates = list(self.dates)
        self.df['Date'] = dates
        self.df.set_index('Date', inplace=True)
        self.df.index = pd.DatetimeIndex(self.df.index)

        fig= plt.figure()
        plt.plot(self.PREDICTIONS_FUTURE.index, self.PREDICTIONS_FUTURE['close'], color='r', label='Predicted Stock Price')
        plt.plot(self.PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, self.PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['close'], color='orange', label='Training predictions')
        plt.plot(self.df.loc[START_DATE_FOR_PLOTTING:].index, self.df.loc[START_DATE_FOR_PLOTTING:]['close'], color='b', label='Actual Stock Price')
        plt.axvline(x = min(self.PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
        plt.grid(which='major', color='#cccccc', alpha=0.5)

        plt.legend(shadow=True)
        plt.title('Predcitions and Actual Stock Prices', family='Arial', fontsize=18)
        plt.xlabel('Timeline', family='Arial', fontsize=14)
        plt.ylabel('Stock Price Value', family='Arial', fontsize=14)
        plt.xticks(rotation=45, fontsize=8)
        st.pyplot(fig)
        