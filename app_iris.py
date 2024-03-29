# import streamlit as st
# import pandas as pd
# import joblib

# st.write(
#     """
# # App Simple pour la prévision des fleurs d'Iris
# Cette application prédit la catégorie des fleurs d'Iris
# """
# )

# st.sidebar.header("Les paramètres d'entrée")


# def user_input():
#     Sepal_Length = st.sidebar.slider("La longueur du Sepal", 4.3, 7.9, 5.3)
#     Sepal_Width = st.sidebar.slider("La largeur du Sepal", 2.0, 4.4, 3.3)
#     Petal_Length = st.sidebar.slider("La longueur du Petal", 1.0, 6.9, 2.3)
#     Petal_Width = st.sidebar.slider("La largeur du Petal", 0.1, 2.5, 1.3)
#     data = {
#         "Sepal.Length": Sepal_Length,
#         "Sepal.Width": Sepal_Width,
#         "Petal.Length": Petal_Length,
#         "Petal.Width": Petal_Width,
#     }
#     fleur_parametres = pd.DataFrame(data, index=[0])
#     return fleur_parametres


# df = user_input()

# st.subheader("On veut trouver la catégorie de cette fleur")
# st.write(df)

# # Charger le modèle avec joblib
# model = joblib.load("C:/Users/MTN Academy/Desktop/Projet_iris/model.pkl")

# # Effectuer la prédiction
# prediction = model.predict(df)

# st.subheader("La catégorie de la fleur d'iris est:")
# st.write(prediction)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.write('''
# App Simple pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')

st.sidebar.header("Les parametres d'entrée")

def user_input():
    Sepal_Length = st.sidebar.slider("La longueur du Sepal", 4.3, 7.9, 5.3)
    Sepal_Width = st.sidebar.slider("La largeur du Sepal", 2.0, 4.4, 3.3)
    Petal_Length = st.sidebar.slider("La longueur du Petal", 1.0, 6.9, 2.3)
    Petal_Width = st.sidebar.slider("La largeur du Petal", 0.1, 2.5, 1.3)
    data = {
        "Sepal.Length": Sepal_Length,
        "Sepal.Width": Sepal_Width,
        "Petal.Length": Petal_Length,
        "Petal.Width": Petal_Width,
    }
    fleur_parametres = pd.DataFrame(data, index=[0])
    return fleur_parametres

data = pd.read_csv('./data/iris.csv')

target = data['Species']
feature = data.drop('Species', axis=1)

#Diviser le dataset pour le jeu d'apprentissage(training) et le test
X_train,X_test,y_train,y_test= train_test_split(feature, target, test_size= 0.2, random_state=40)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

df=user_input()

st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)


prediction=model.predict(df)

st.subheader("La catégorie de la fleur d'iris est:")
st.write(prediction)
