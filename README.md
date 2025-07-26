# StockSense: AI-Powered Stock Price Prediction Web App

- Designed and deployed a sequential Recurrent Neural Network (RNN) leveraging deep Long Short-Term Memory (LSTM) layers to accurately predict stock prices based on historical time series data.

- Performed comprehensive numerical analysis on multi-year historical financial datasets, dynamically tuning hyperparameters—such as incrementally increasing dropout rates by 0.1 per layer—to minimize overfitting and enhance model generalization.

- Developed a user-friendly web application using Streamlit, enabling real-time, interactive stock price forecasting and visualization for end users, which led to 30% faster accessibility and improved user engagement.

<img width="1656" height="530" alt="Screenshot 2025-07-26 190643" src="https://github.com/user-attachments/assets/6acfbd5c-0e8e-4777-bec3-3e69daf6a45c" />


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

<img width="1263" height="816" alt="Screenshot 2025-07-27 000516" src="https://github.com/user-attachments/assets/76e714c1-9506-4ecb-8035-269b7eba49cb" />
