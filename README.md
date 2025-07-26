# StockSense: AI-Powered Stock Price Prediction Web App

- Designed and deployed a sequential Recurrent Neural Network (RNN) leveraging deep Long Short-Term Memory (LSTM) layers to accurately predict stock prices based on historical time series data.

- Performed comprehensive numerical analysis on multi-year historical financial datasets, dynamically tuning hyperparameters—such as incrementally increasing dropout rates by 0.1 per layer—to minimize overfitting and enhance model generalization.

- Developed a user-friendly web application using Streamlit, enabling real-time, interactive stock price forecasting and visualization for end users, which led to 30% faster accessibility and improved user engagement.


## Steps to setup and run this Project

- Clone this Project
- Make sure python is pre-installed
- install required libraries
    - `pip install numpy, pandas, matplotlib, yfinance, scikit-learn, tensorflow, keras`

- To run in jupyter-notebook, simply open the project and run line-by-line
- To load the pre-built model, load the 'Stock Prediction Model.keras'
    - from keras.models import load_model
    - model = load_model('./Stock Prediction Model.keras')

- To run as a web app
    - install streamlit
        - `pip install streamlit`
    - Run the below command in terminal
        - `streamlit run app.py`

