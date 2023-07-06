import streamlit as st

# Dictionnaire pour stocker les informations d'inscription (exemple simple)
users = {}

def main():
    st.title("Application d'authentification")

    # Afficher la page d'accueil avec les boutons de connexion et d'inscription
    page = st.sidebar.radio("Navigation", ["Accueil", "Connexion", "Inscription"])

    if page == "Accueil":
        st.write("Bienvenue dans l'application !")
        st.write("Veuillez choisir une option dans la barre latérale.")

    elif page == "Connexion":
        login()

    elif page == "Inscription":
        register()

def register():
    st.title("Inscription")
    email = st.text_input("E-mail")
    password = st.text_input("Mot de passe", type="password")

    if st.button("S'inscrire"):
        # Vérifier si l'utilisateur existe déjà
        if email in users:
            st.error("Cet e-mail est déjà utilisé.")
        else:
            # Stocker les informations d'inscription dans le dictionnaire
            users[email] = password
            st.success("Inscription réussie ! Veuillez vous connecter.")
            # Afficher le formulaire de connexion après l'inscription réussie
            login()

def login():
    st.title("Connexion")
    email = st.text_input("E-mail")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        # Vérifier les informations d'authentification
        if email in users and users[email] == password:
            st.success("Connexion réussie.")
            # Afficher le contenu de l'application après la connexion réussie
            show_app_content()
        else:
            st.error("E-mail ou mot de passe incorrect.")

def show_app_content():
    st.title("Application")
    st.write("Bienvenue dans l'application !")
    st.write("Vous êtes connecté.")

    # Afficher les options de l'application
    option = st.selectbox("Que souhaitez-vous faire ?", ["Importer un fichier Excel", "Se déconnecter"])

    if option == "Importer un fichier Excel":
        # Ajoutez ici votre code pour importer un fichier Excel
        st.write("Vous avez choisi d'importer un fichier Excel.")
    elif option == "Se déconnecter":
        # Afficher le formulaire de connexion pour permettre à l'utilisateur de se déconnecter
        login()

# Appeler la fonction principale
if __name__ == "__main__":
    main()
