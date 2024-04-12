import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')


def preprocess_data(df):
   
    df.dropna(subset=['Name','Year','Duration','Votes','Rating'], inplace=True)
   
    df['Year'] = df['Year'].str.strip('()').astype(int)
    
    df['Votes'] = df['Votes'].str.replace(',', '').astype(int)
   
    df['Duration'] = df['Duration'].str.replace('min', '').astype(int)
    
    df.drop(['Name','Director','Actor 1','Actor 2','Actor 3'], axis=1, inplace=True)
    return df


def train_or_load_model(df):
    try:
        model = joblib.load("imdb_rating_model.pkl")
        st.write("Loaded pre-trained model.")
        X = df[['Year','Duration','Votes']]
        y = df['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
    except FileNotFoundError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('sgd', SGDRegressor(max_iter=10000, random_state=1000))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, "imdb_rating_model.pkl")
        st.write("Trained and saved model.")
        model = pipeline
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2


preprocessed_df = preprocess_data(df)
model, X_test, y_test = train_or_load_model(preprocessed_df)


def main(X_test, y_test):
    st.title('IMDb Movie Rating Prediction')

    st.sidebar.header('Input Parameters')
    year = st.sidebar.number_input('Year', value=2023)
    duration = st.sidebar.number_input('Duration (minutes)', value=120)
    votes = st.sidebar.number_input('Votes', value=10000)

    new_input = pd.DataFrame({
        'Year': [year],
        'Duration': [duration],
        'Votes': [votes]
    })

    if st.sidebar.button('Predict'):
        predicted_rating = model.predict(new_input)
        st.write(f'Predicted Rating: {predicted_rating[0]}')


if __name__ == '__main__':
    main(X_test, y_test)
