import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
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
from openpyxl import reader, load_workbook, Workbook
import pickle

# Charger le modèle depuis le fichier
def load_model():
    with open('import.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def m(data):
    mois = data[0]
    if mois == 'janvier':
        return 1
    if mois == 'février':
        return 2
    if mois == 'mars':
        return 3
    if mois == 'avril':
        return 4
    if mois == 'mai':
        return 5
    if mois == 'juin':
        return 6
    if mois == 'juillet':
        return 7
    if mois == 'août':
        return 8
    if mois == 'septembre':
        return 9
    if mois == 'octobre':
        return 10
    if mois == 'novembre':
        return 11
    if mois == 'décembre':
        return 12


# Store the original month names
month_mapping = {
    1: 'janvier',
    2: 'février',
    3: 'mars',
    4: 'avril',
    5: 'mai',
    6: 'juin',
    7: 'juillet',
    8: 'août',
    9: 'septembre',
    10: 'octobre',
    11: 'novembre',
    12: 'décembre'
}


# Fonction de prédiction
def prediction(data):
    # Prétraitement des données (assurez-vous d'avoir les mêmes étapes de prétraitement que lors de l'entraînement)
    # ...

    # Effectuer la prédiction
    dmatrix = xgb.DMatrix(data)
    model = load_model()
    predictions = model.predict(dmatrix)

    # Arrondir les prédictions au nombre entier supérieur
    rounded_predictions = np.ceil(predictions).astype(int)

    # Remplacer les prédictions nulles ou négatives par 1
    rounded_predictions[rounded_predictions <= 0] = 1

    # Map the predicted month values to the original month names
    predicted_months = pd.DataFrame(data['Mois'].map(month_mapping), columns=['Mois'])

    # Combine the predicted months with the rounded predictions
    result = pd.concat([predicted_months, pd.DataFrame(rounded_predictions, columns=['Predictions'])], axis=1)

    return result


def pretraitement_data(data):
    # Code pour prétraiter les données
    # ...

    # Suppression des colonnes spécifiées
    columns_to_drop = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature', 'Trimestre']

    dataset = data.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()
    dataset['Unité_Commande'] = encoder.fit_transform(dataset['Unité_Commande'])
    dataset['Article'] = encoder.fit_transform(dataset['Article'])

    dataset['Mois'] = dataset[['Mois']].apply(m, axis=1)

    # Remplacer les valeurs nulles par la moyenne
    mean_time = dataset["Tps_dappro"].mean()
    dataset["Tps_dappro"].fillna(mean_time, inplace=True)

    # Retourner les colonnes originales du DataFrame
    original_columns = ['Mois', 'Unité_Commande', 'Article', 'Tps_dappro']
    dataset = dataset[original_columns]

    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    return dataset

# Streamlit app code
st.title('Prédiction des commandes')
st.write('')

# Importer les données à partir d'un fichier Excel
uploaded_file = st.file_uploader("Importer le fichier Excel contenant les données", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.write('**Aperçu des données importées :**')
        st.write(data.head())
        st.write('')

        # Prétraitement des données importées
        processed_data = pretraitement_data(data)

        # Prédiction sur les données importées
        predictions = prediction(processed_data)

        st.write('**Résultats des prédictions :**')
        st.write(predictions)
        st.write('')

        # Enregistrer les résultats dans un fichier Excel
        predictions_filename = 'predictions.xlsx'
        predictions.to_excel(predictions_filename, index=False)
        st.markdown(f'**[Télécharger les résultats des prédictions]({predictions_filename})**')
    except Exception as e:
        st.write('Une erreur s\'est produite lors du chargement du fichier.')
        st.write(e)