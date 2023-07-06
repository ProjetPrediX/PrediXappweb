import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import xgboost as xgb
#import pickle


def main():
    # ... Code de configuration Streamlit ...
    os.chdir("C:\\Users\\LENOVO\\OneDrive\\Bureau\\pythonProject11")
    dataset = pd.read_excel("Local.xlsx")
    # Affichage du DataFrame
    st.write(dataset)


    # Préparer les données pour XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Définir les paramètres du modèle de gradient boosting
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.2,  # Taux d'apprentissage
        'max_depth': 5,  # Profondeur maximale de l'arbre
        'subsample': 0.8,  # Proportion d'échantillons utilisés pour la construction de chaque arbre
        'colsample_bytree': 0.8,  # Proportion de caractéristiques utilisées pour la construction de chaque arbre
        'seed': 42  # Graine aléatoire pour la reproductibilité
    }
    global model
    # Entraîner le modèle sur les données d'entraînement
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    st.write("Prédiction de X_test")
    # Prédire sur les données de test
    y_pred = model.predict(dtest)
    y_pred = np.round(y_pred).astype(int)
    st.write(y_pred)



    # Importer un fichier Excel
    file = st.file_uploader("Importer un fichier Excel", type=["xlsx"], key="file_uploader")


    # Vérifier si un fichier a été importé
    if file is not None:
        # Lire le fichier Excel en tant que DataFrame
        df = pd.read_excel(file)

        # Afficher le DataFrame importé
        st.write("Données importées :")
        st.write(df)

        # Bouton pour effectuer les prédictions
        if st.button("Effectuer les prédictions",  key="predict_button"):
            # Nettoyage des données
            df_cleaned = df.drop(columns_to_drop1, axis=1)
            encoder = LabelEncoder()
            encoder.fit(df_cleaned['Unité_Commande'])
            df_cleaned['Unité_Commande'] = encoder.transform(df_cleaned['Unité_Commande'])
            encoder.fit(df_cleaned['Article'])
            df_cleaned['Article'] = encoder.transform(df_cleaned['Article'])
            df_cleaned['Mois'] = df_cleaned[['Mois']].apply(m, axis=1)
            df_cleaned['Tps_d_appro'] = df_cleaned[['Tps_d_appro']].apply(moy, axis=1)

            # Effectuer les prédictions
            dmatrix_pred = xgb.DMatrix(df_cleaned)
            predictions = model.predict(dmatrix_pred)
            rounded_predictions = np.round(predictions).astype(int)

            # Ajouter les prédictions au DataFrame
            df_cleaned['Prédictions'] = rounded_predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df_cleaned)

# Appeler la fonction principale
if __name__ == "__main__":
    main()
