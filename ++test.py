import streamlit as st
from login import login
from signup import signup
from logout import logout
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
from login import login
from signup import signup
from auth import auth


def home():
    st.write ('Bienvenue sur le site de mon projet Predix !')


def main():
    st.set_page_config (page_title='Projet Predix')
    st.title ('Projet Predix')

    # Vérifier si l'utilisateur est déjà connecté
    if not st.session_state.get ('username'):
        home ( )  # Page d'accueil
        signup ( )  # Page d'inscription
        login ( )  # Page de connexion
    else:
        # Afficher la page principale de l'application
        st.write ('Bienvenue, {} !'.format (st.session_state['username']))
        # ...
        # Vos étapes principales de l'application ici

    # Barre de navigation
    nav = st.sidebar.radio ('Navigation', ['Accueil', 'Déconnexion'])

    if nav == 'Accueil':
        home ( )
    elif nav == 'Déconnexion':
        logout ( )


if __name__ == '__main__':
    main ( )