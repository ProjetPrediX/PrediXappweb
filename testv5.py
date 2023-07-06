from streamlit_option_menu import option_menu
from login import login
from signup import signup
from logout import logout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
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
import nbformat as nbf
from nbconvert import MarkdownExporter

def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model
model_path_import = "C:\\Users\\hp\\Desktop\\modele.xgb"
model_import = xgb.Booster()
model_import.load_model(model_path_import)

# Charger le modèle pré-entrainé pour l'achat local
model_path_local = "C:\\Users\\hp\\Desktop\\modele.xgb1"
model_local = xgb.Booster()
model_local.load_model(model_path_local)



def perform_predictions_import(data):
    # Preprocessing steps for import
    # ...
    # Use model_import to perform predictions
    predictions = model_import.predict(data)
    return predictions

def perform_predictions_local(data):
    # Preprocessing steps for local
    # ...
    # Use model_local to perform predictions
    predictions = model_local.predict(data)
    return predictions

def home():
    st.write('Bienvenue sur le site de mon projet Predix !')

def perform_predictions(data):
    # Prétraitement des données (adapté à votre pipeline de prétraitement)
    # ...

    # Convertir les données en une matrice DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(data)
    global model
    # Effectuer les prédictions
    predictions = model.predict(dmatrix)

    # Retourner les prédictions arrondies
    return np.round(predictions)


def convert_ipynb_to_markdown(ipynb_file):
    with open(ipynb_file, "r") as f:
        notebook = nbf.read(f, as_version=4)

    exporter = MarkdownExporter()
    markdown, _ = exporter.from_notebook_node(notebook)

    return markdown



def preprocess_data(data):
    # Perform any necessary preprocessing steps here
    encoder = LabelEncoder()
    encoder.fit(data['Unité_Commande'])
    data['Unité_Commande'] = encoder.transform(data['Unité_Commande'])
    encoder.fit(data['Article'])
    data['Article'] = encoder.transform(data['Article'])
    data['Mois'] = data['Mois'].apply(lambda x: int(x.split('-')[1]))
    data['Tps_dappro'] = data['Tps_dappro'].apply(lambda x: np.mean(list(map(int, x.split(',')))))
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data

def main():
    st.set_page_config(page_title='Projet Predix')
    st.title('Projet Predix')

    # Vérifier si l'utilisateur est déjà connecté
    if not st.session_state.get('username'):
        home()  # Page d'accueil
        signup()  # Page d'inscription
        login()  # Page de connexion
    else:
        # Afficher la page principale de l'application
        st.write('Bienvenue, {} !'.format(st.session_state['username']))
        # ...
        # Vos étapes principales de l'application ici

    # Interface Streamlit
    st.title("Prédictions à partir d'un fichier Excel")

    # Importer un fichier Excel
    file = st.file_uploader("Importer un fichier Excel", type=["xlsx"], key="file_uploader")
    columns_to_drop1 = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total',
                        'Nom_fournisseur',
                        'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                        'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                        'Trimestre', 'Qté_Commandé']

    # Vérifier si un fichier a été importé
    if file is not None:
        # Lire le fichier Excel en tant que DataFrame
        df = pd.read_excel(file)

        # Afficher le DataFrame importé
        st.write("Données importées :")
        st.write(df)

        # Bouton pour effectuer les prédictions
        if st.button("Effectuer les prédictions", key="predict_button"):
            columns_to_drop1 = [col for col in columns_to_drop1 if col in df.columns]
            # Nettoyage des données
            df_cleaned = df.drop(columns_to_drop1, axis=1)
            encoder = LabelEncoder()
            encoder.fit(df_cleaned['Unité_Commande'])
            df_cleaned['Unité_Commande'] = encoder.transform(df_cleaned['Unité_Commande'])
            encoder.fit(df_cleaned['Article'])
            df_cleaned['Article'] = encoder.transform(df_cleaned['Article'])
            df_cleaned['Mois'] = df_cleaned['Mois'].apply(lambda x: int(x.split('-')[1]))
            df_cleaned['Tps_dappro'] = df_cleaned['Tps_dappro'].apply(lambda x: np.mean(list(map(int, x.split(',')))))
            scaler = MinMaxScaler()
            df_cleaned1 = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

            # Charger le modèle pré-entrainé
            model_path = "C:\\Users\\hp\\Desktop\\model.xgb"  # Chemin vers votre modèle pré-entrainé
            model = xgb.Booster()
            model.load_model(model_path)

            # Effectuer les prédictions
            dmatrix_pred = xgb.DMatrix(df_cleaned1)
            predictions = model.predict(dmatrix_pred)
            rounded_predictions = np.round(predictions).astype(int)

            # Ajouter les prédictions au DataFrame
            df_cleaned['Prédictions'] = rounded_predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df_cleaned)

    # Barre de navigation
    nav = st.sidebar.radio('Navigation', ['Accueil', 'Déconnexion'])
    # 1. as sidebar menu
    with st.sidebar:
        selected = option_menu("Menu principal", ["Accueil", "Achat Import", "Achat Local", "Paramètres"],
                               icons=['house', 'cloud-upload', 'chart-line', 'gear'],
                               menu_icon="cast", default_index=0)

    # 2. Contenu de la page Accueil
        if selected == "Accueil":
            st.write("Bienvenue sur la page d'accueil !")

    # 3. Contenu de la page Achat Import
        elif selected == "Achat Import":
            st.write("Page Achat Import - Prédictions liées à l'achat import")
            data = preprocess_data(df_cleaned1)  # Preprocess the data for import
            predictions = perform_predictions_import(data)  # Perform predictions using the import model
            st.write(predictions)  # Display the predictions
            # markdown_content = convert_ipynb_to_markdown("C:\\Users\\hp\\Desktop\\import.ipynb")
            # st.markdown(markdown_content)

        elif selected == "Achat Local":
            st.write("Page Achat Local - Prédictions liées à l'achat local")
            data = preprocess_data(df_cleaned1)  # Preprocess the data for local
            predictions = perform_predictions_local(data)  # Perform predictions using the local model
            st.write(predictions)  # Display the predictions

    # 4. Contenu de la page Achat Local







        if nav == 'Accueil':
            home()
        elif nav == 'Déconnexion':
            logout()


if __name__ == '_main_':
    main()