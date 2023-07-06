import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import xgboost as xgb
from openpyxl import reader,load_workbook,Workbook
from sklearn.metrics import mean_squared_error, r2_score


st.write('''
# L'intelligence artificielle à votre service
PrediX
''')


st.write('''
data avant nettoyage
''')

def afficher_graphique():
    x = np.arange(len(y_test))
    fig, ax = plt.subplots()  # Créer une figure et un axe
    # Prédictions

    ax.plot(x, y_test, label='Données réelles')
    ax.plot(x, y_pred, label='Prédictions')
    ax.legend()
    st.pyplot(fig)  # Utiliser st.pyplot() pour afficher le graphique


# kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk




def perform_predictions(data):
    # Prétraitement des données (adapté à votre pipeline de prétraitement)
    # ...

    # Convertir les données en une matrice DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(data)

    # Effectuer les prédictions
    predictions = model.predict(dmatrix)

    # Retourner les prédictions arrondies
    return np.round(predictions)




def main():
    # ... Code de configuration Streamlit ...
    os.chdir("C:\\Users\\LENOVO\\OneDrive\\Bureau\\pythonProject11")
    dataset = pd.read_excel("Local.xlsx")
    # Affichage du DataFrame
    st.write(dataset)

    # Matrice de corrélation
    # corr = dataset.corr()
    # plt.figure(figsize=(12, 9))
    # sns.heatmap(corr, annot=True, cmap='coolwarm')
    # plt.show()
    # st.pyplot(plt.gcf())

    # Suppression des colonnes spécifiées
    columns_to_drop = ['Qté_Récep.', 'Unité_Récep.', 'Fournisseur', 'CC(O/N)', 'Jour', 'MT total', 'Nom_fournisseur',
                       'Désignation', 'N° BC', 'N° BL', 'Qté', 'Unité', 'Montant', 'Type', 'Coût_unitaire_moyen',
                       'Réglement', 'Année', 'Prix_unitaire', 'Unite_de_prix', 'Code_Nature',
                       'Trimestre']

    dataset = dataset.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    Article = dataset['Article']
    Unité_Commande = dataset['Unité_Commande']

    # Créer une instance de l'encodeur de label
    encoder = LabelEncoder()

    # Encoder les valeurs de la colonne 'Unité de reception'
    recep_encoded = encoder.fit_transform(dataset['Unité_Commande'])
    article_encoded = encoder.fit_transform(dataset['Article'])

    # Remplacer les colonnes d'unité de prix et d'unité de réception par les colonnes encodées
    dataset['Unité_Commande'] = recep_encoded
    dataset['Article'] = article_encoded

    Article = dataset['Article']

    Unité_Commande = dataset['Unité_Commande']

    # Créer une instance de l'encodeur de label
    encoder = LabelEncoder()

    # Encoder les valeurs de la colonne 'Unité de reception'
    recep_encoded = encoder.fit_transform(dataset['Unité_Commande'])
    article_encoded = encoder.fit_transform(dataset['Article'])

    # Remplacer les colonnes d'unité de prix et d'unité de réception par les colonnes encodées
    dataset['Unité_Commande'] = recep_encoded
    dataset['Article'] = article_encoded

    # Remplacer les colonnes d'unité de prix et d'unité de réception par les colonnes encodées
    dataset['Unité_Commande'] = recep_encoded
    dataset['Article'] = article_encoded

    article = pd.concat([Article, dataset['Article']], axis=1, keys=['Article', 'Article_code'])
    commande = pd.concat([Unité_Commande, dataset['Unité_Commande']], axis=1,
                         keys=['Unité_Commande', 'Unité_Commande_code'])

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

    dataset['Mois'] = dataset[['Mois']].apply(m, axis=1)

    global moyenne
    moyenne = dataset['Tps_d_appro'].mean()

    def moy(data):
        Tps_Réappro = data[0]
        if pd.isnull(Tps_Réappro):
            return moyenne
        else:
            return Tps_Réappro

    dataset['Tps_d_appro'] = dataset[['Tps_d_appro']].apply(moy, axis=1)

    st.write('''
    data après nettoyage
    ''')
    X = dataset.iloc[:, [0, 1, 2, 4]]
    y = dataset.iloc[:, [3]]

    st.write(dataset)
    # Écrire les données dans un fichier Excel
    output_file1 = "predict.xlsx"
    X.to_excel(output_file1, index=False)
    st.write(f"Données écrites dans le fichier Excel : {output_file1}")

    # Utiliser la méthode MinMaxScaler pour normaliser les données
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    global y_test, y_pred, model
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
    st.write("Prediction de X_test")
    # Prédire sur les données de test
    y_pred = model.predict(dtest)
    y_pred = y_pred.round().astype(int)
    st.write(y_pred)
    # Évaluer les performances du modèle en utilisant le RMSE, MAE et R²
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("Performance sur l'ensemble de test :")
    st.write("rmse :")

    st.write(rmse)
    st.write("r2 :")
    st.write(r2)
    st.write("mae :")
    st.write(mae)

    # Données réelles
    st.write('Réel vs prédiction')
    # Charger les données depuis le fichier Excel


    # ... Code Streamlit pour l'interface utilisateur ...

    # Bouton pour afficher le graphique
    if st.button("Afficher le graphique"):
        afficher_graphique()

    # Interface Streamlit
    st.title("Prédictions à partir d'un fichier Excel")

    # Importer un fichier Excel
    file = st.file_uploader("Importer un fichier Excel", type=["xlsx"])

    # Vérifier si un fichier a été importé
    if file is not None:
        # Lire le fichier Excel en tant que DataFrame
        df = pd.read_excel(file)

        # Afficher le DataFrame importé
        st.write("Données importées :")
        st.write(df)

        # Bouton pour effectuer les prédictions
        if st.button("Effectuer les prédictions"):
            # Effectuer les prédictions
            predictions = perform_predictions(df)
            # Ajouter les prédictions au DataFrame
            df["Prédictions"] = predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df)
# Appeler la fonction principale
if __name__ == "__main__":
    main()