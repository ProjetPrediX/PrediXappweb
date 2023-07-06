import streamlit as st
from models.import_model import perform_predictions_import
from utils.data_preprocessing import pretraitement_data
import pandas as pd

def import_prediction_page():
    st.subheader("Prédiction d'achats - Import")
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
            pretraitement_data(df)
            predictions = perform_predictions_import(df)  # Perform predictions using the import model
            st.write(predictions)
    # Code pour importer le fichier Excel
    # ...

    # Code pour prétraiter les données
    # ...

    # Code pour effectuer les prédictions
    # ...

    # Affichage des résultats
    # ...
