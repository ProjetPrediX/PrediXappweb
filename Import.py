import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import base64
#import seaborn as sns
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



def pretraitement_data1(data):
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

    # Conversion des mois en entiers
    mois_mapping = {
        'janvier': 1,
        'février': 2,
        'mars': 3,
        'avril': 4,
        'mai': 5,
        'juin': 6,
        'juillet': 7,
        'août': 8,
        'septembre': 9,
        'octobre': 10,
        'novembre': 11,
        'décembre': 12
    }
    dataset['Mois'] = dataset['Mois'].map(mois_mapping)

    # Remplacement des valeurs nulles dans la colonne "Tps_dappro" par la moyenne
    mean_time = dataset["Tps_dappro"].mean()
    dataset["Tps_dappro"].fillna(mean_time, inplace=True)

    # Mise à l'échelle des données
    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    return dataset


# Fonction de prédiction
def prediction(data):
    # Prétraitement des données (assurez-vous d'avoir les mêmes étapes de prétraitement que lors de l'entraînement)
    # ...


    # Effectuer la prédiction
    dmatrix = xgb.DMatrix(data)
    model=load_model()
    predictions = model.predict(dmatrix)

    return predictions

def pretraitement_data(data):
    # Code pour prétraiter les données
    # ...


    # Code pour effectuer les prédictions avec le modèle local
    # ...
    # Suppression des colonnes spécifiées
    columns_to_drop = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                       'Trimestre']

    dataset = data.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()

    dataset['Unité_Commande'] = encoder.fit_transform(dataset['Unité_Commande'])
    dataset['Article'] = encoder.fit_transform(dataset['Article'])

    dataset['Mois'] = dataset[['Mois']].apply(m, axis=1)
    dataset['Mois']=np.round(dataset['Mois']).astype(int)
    mean_time = dataset["Tps_dappro"].mean()

    # Remplacer les valeurs nulles par la moyenne
    dataset["Tps_dappro"].fillna(mean_time, inplace=True)

    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    return dataset



# Interface utilisateur avec Streamlit
def main():
    # Titre de l'application
    st.title("Application de prédiction")
    st.title("VISUALISATION REELLE PAR RAPPORT A LA PREDICTION avec le modele XGBOOST")
    if st.button("Afficher le graphique"):
        # Chemin du graphique enregistré
        graphique_path = 'graphique_import.png'

        # Enregistrer le chemin du graphique dans un fichier pickle
        with open('graphique_import.pkl', 'wb') as file:
            pickle.dump(graphique_path, file)

        st.image(graphique_path)

    # Formulaire de téléchargement du fichier Excel
    st.subheader("Télécharger le fichier Excel contenant les nouvelles données")

    # Vérifier si un fichier a été téléchargé
    file = st.file_uploader("Importer un fichier Excel", type=["xlsx"], key="file_uploader")

    #columns_to_drop1 = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                        # 'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                        # 'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                        # 'Trimestre', 'Qté_Commandé']

    # Vérifier si un fichier a été importé
    if file is not None:
        # Lire le fichier Excel en tant que DataFrame
        df = pd.read_excel(file)

        # Afficher le DataFrame importé
        st.write("Données importées :")
        st.write(df)

        # Bouton pour effectuer les prédictions
        if st.button("Effectuer les prédictions", key="predict_button"):

            # Effectuer les prédictions
            df_cleaned=pretraitement_data(df)
            predictions = prediction(df_cleaned)
            rounded_predictions = np.round(predictions).astype(int)

            # Ajouter les prédictions au DataFrame
            df['Prédiction Qté commandée'] = rounded_predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df)
            # Convert the DataFrame to Excel file
            excel_path = "predictions.xlsx"
            df_cleaned.to_excel(excel_path, index=False)

            # Encode the Excel file data to Base64
            with open(excel_path, "rb") as file:
                excel_data = file.read()
            b64 = base64.b64encode(excel_data).decode()

            # Generate the download link
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx">Télécharger le fichier Excel</a>'
            st.markdown(href, unsafe_allow_html=True)


# Appel de la fonction principale pour exécuter l'application
if __name__ == "__main__":
    main()
