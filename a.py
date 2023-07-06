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


st.write('''
# L'intelligence artificielle à votre service
PrediX
''')

st.write('''
data avant nettoyage
''')
global y_test
global y_pred
global model
def pretraitement_data(data):
    # Suppression des colonnes spécifiées
    columns_to_drop = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                       'Trimestre']

    data = data.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()
    data['Unité_Commande'] = encoder.fit_transform(data['Unité_Commande'])
    data['Article'] = encoder.fit_transform(data['Article'])

    def map_mois(mois):
        mois_dict = {'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
                     'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12}
        return mois_dict.get(mois.lower())

    data['Mois'] = data['Mois'].map(map_mois)

    # Remplacement des valeurs manquantes dans Tps_d_appro par la moyenne
    moyenne = data['Tps_d_appro'].mean()
    data['Tps_d_appro'].fillna(moyenne, inplace=True)

    return data


def afficher_graphique():
    global y_test
    global y_pred
    global model
    x = np.arange(len(y_test))
    fig, ax = plt.subplots()  # Créer une figure et un axe
    # Prédictions

    ax.plot(x, y_test, label='Données réelles')
    ax.plot(x, y_pred, label='Prédictions')
    ax.legend()
    st.pyplot(fig)  # Utiliser st.pyplot() pour afficher le graphique


def entrainement(data):
    global y_test
    global y_pred
    global model

    # Prétraitement des données (adapté à votre pipeline de prétraitement)
    # ...

    X = data.iloc[:, [0, 1, 2, 4]]
    y = data.iloc[:, [3]]

    # Utiliser la méthode MinMaxScaler pour normaliser les données
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

    # Entraîner le modèle sur les données d'entraînement
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    st.write("Prédiction de X_test")
    # Prédire sur les données de test
    y_pred = model.predict(dtest)
    y_pred = np.round(y_pred).astype(int)
    st.write(y_pred)


def pred(data):
    global y_test
    global y_pred
    global model

    # Convertir les données en une matrice DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(data)

    # Effectuer les prédictions
    predictions = model.predict(dmatrix)

    # Retourner les prédictions arrondies
    return predictions.round().astype(int)

def main():

    # ... Code de configuration Streamlit ...
    os.chdir("C:\\Users\\LENOVO\\OneDrive\\Bureau\\pythonProject11")
    dataset = pd.read_excel("Local.xlsx")
    # Affichage du DataFrame
    st.write(dataset)
    pretraitement_data(dataset)
    entrainement(dataset)
    st.write('''
    data après nettoyage
    ''')
    st.write(dataset)
    global y_test
    global y_pred
    global model

    # Évaluer les performances du modèle en utilisant le RMSE, MAE et R²
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("Performance sur l'ensemble de test :")
    st.write("rmse :", rmse)
    st.write("r2 :", r2)
    st.write("mae :", mae)

    # Données réelles
    st.write('Réel vs prédiction')

    # Bouton pour afficher le graphique
    if st.button("Afficher le graphique"):
        afficher_graphique()

    # Interface Streamlit
    st.title("Prédictions à partir d'un fichier Excel")


    # Importer un fichier Excel
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
        if st.button("Effectuer les prédictions",  key="predict_button"):
            # Nettoyage des données
            # Effectuer les prédictions

            # Ajouter les prédictions au DataFrame
            df['Prédictions'] = pred(df)

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df)

# Appeler la fonction principale
if __name__ == "__main__":
    main()
