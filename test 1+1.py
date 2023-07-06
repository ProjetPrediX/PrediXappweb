from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb

model_path_local = "C:\\Users\\hp\\Desktop\\modele.xgb1"
model_local = xgb.Booster()
model_local.load_model(model_path_local)

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

    dataset = dataset.drop(columns_to_drop, axis=1)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()

    dataset['Unité_Commande'] = encoder.fit_transform(dataset['Unité_Commande'])
    dataset['Article'] = encoder.fit_transform(dataset['Article'])

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
    moyenne = dataset['Tps_dappro'].mean()

    def moy(data):
        Tps_Réappro = data[0]
        if pd.isnull(Tps_Réappro):
            return moyenne
        else:
            return Tps_Réappro

    dataset['Tps_dappro'] = dataset[['Tps_dappro']].apply(moy, axis=1)

    st.write('''
        data après nettoyage
        ''')
    st.write(dataset)

    X = dataset.iloc[:, [0, 1, 2, 4]]
    y = dataset.iloc[:, [3]]
    global y_test
    global y_pred
    global model
    # Utiliser la méthode MinMaxScaler pour normaliser les données
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Préparer les données pour XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return data
