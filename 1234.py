from Import import pretraitement_data,prediction

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
            df_cleaned['Prédiction Qté commandée'] = rounded_predictions

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(df_cleaned)
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
