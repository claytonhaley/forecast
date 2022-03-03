import os
import shutil
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import ModelIntegration


        
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

        lookback_days = st.sidebar.slider("Past days considered for training", min_value=1, max_value=730)

        lookahead_days = st.sidebar.slider("Days ahead to predict", min_value=1, max_value=100)

        epochs = st.sidebar.slider("Epochs", min_value=1, max_value=100)

        submit = st.form_submit_button('Run Model')
        tune_choice = st.checkbox('Hyperparameter Tuning')
        tuner = False

        if tune_choice:
            tuner = True

        if submit:
            try:
            # ---- DATA PREPROCESSING ----
                integrate = ModelIntegration(ticker_name, start_date, end_date, lookback_days, lookahead_days, epochs)

                integrate._preprocess()

                integrate._create_dataset()
            
                # Instantiate Model Architecture
                with st.spinner('Training Model...'):
                    if tuner == False:
                        model, history = integrate._build_default_model()
                    else:
                        if os.path.isdir('untitled_project'): shutil.rmtree('untitled_project')

                        st.warning("This may take some time.")

                        model, history = integrate._run_tuned_model()

                
                with left_column:
                    fig = plt.figure()
                    plt.plot(history.history['loss'], label='Training loss')
                    plt.plot(history.history['val_loss'], label='Validation loss')
                    plt.legend()
                    st.pyplot(fig)

                # Generating Predictions
                with st.spinner('Generating Predictions...'):
                    rmse, r2, future_predictions, train_predictions = integrate._generate_predictions(model)
                    
                    with right_column:
                        integrate._pred_plot()

                st.success('Done!')

                # Generating RMSE
                with left_column:
                    st.warning(f"Stopped at {len(history.history['loss'])} epochs (loss failed to improve)")
                    st.info(f"RMSE: {rmse}")
                    st.info("R-Squared: " + "{:.2%}".format(r2))

                    st.dataframe(future_predictions)
            
            except:
                st.error("Please choose the appropriate parameters")


if __name__ == "__main__":
    main()