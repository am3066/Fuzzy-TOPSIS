
import streamlit as st
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION DE L'APPLICATION
# =====================================================

st.set_page_config(
    page_title="Fuzzy TOPSIS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DESIGN BABY PINK - CONTRASTE AMELIORE
# =====================================================


st.markdown("""
<style>

/* Fond principal */
.stApp {
    background-color: #FFF8FB;
    color: #432338;
}

/* Texte et titres */
.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp p, .stApp label {
    color: #432338 !important;
}

.stApp h1 {
    color: #853F62 !important;
}

/* Barre laterale */
section[data-testid="stSidebar"] {
    background-color: #F6D6E4 !important;
}

section[data-testid="stSidebar"] * {
    color: #432338 !important;
}

/* Tous les champs de saisie */
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="textarea"] {
    background-color: #FFFFFF !important;
    color: #432338 !important;
    border: 1px solid #D58BAE !important;
    border-radius: 8px !important;
}

/* Texte des listes deroulantes */
div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
div[data-baseweb="textarea"] * {
    color: #432338 !important;
    -webkit-text-fill-color: #432338 !important;
}

/* Fond de la liste des options */
ul[role="listbox"],
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #432338 !important;
}

/* Options survolees */
li[role="option"]:hover {
    background-color: #F6D6E4 !important;
}

/* Sections deployables */
div[data-testid="stExpander"] {
    background-color: #FFF0F6 !important;
    border: 1px solid #D58BAE !important;
    border-radius: 10px !important;
}

/* En-tete des sections */
div[data-testid="stExpander"] summary {
    background-color: #F6D6E4 !important;
    border-radius: 8px !important;
}

/* Texte des sections */
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] summary p {
    color: #432338 !important;
    -webkit-text-fill-color: #432338 !important;
}

/* Boutons */
.stButton > button,
.stDownloadButton > button {
    background-color: #A64270 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

.stButton > button * ,
.stDownloadButton > button * {
    color: white !important;
}

.stButton > button:hover {
    background-color: #853F62 !important;
}

/* Champs numeriques */
div[data-testid="stNumberInput"] button {
    background-color: #F6D6E4 !important;
}

/* Indicateurs */
div[data-testid="stMetric"] {
    background-color: #FCE5EF;
    padding: 15px;
    border-radius: 10px;
}

/* Separateurs */
hr {
    border-color: #E5ADC7 !important;
}

</style>
""", unsafe_allow_html=True)



# =====================================================
# ECHELLES LINGUISTIQUES
# =====================================================

notes = {
    "VP": (0, 0, 1),
    "P": (0, 1, 3),
    "MP": (1, 3, 5),
    "F": (3, 5, 7),
    "MG": (5, 7, 9),
    "G": (7, 9, 10),
    "VG": (9, 10, 10)
}

poids = {
    "VL": (0, 0, 0.1),
    "L": (0, 0.1, 0.3),
    "ML": (0.1, 0.3, 0.5),
    "M": (0.3, 0.5, 0.7),
    "MH": (0.5, 0.7, 0.9),
    "H": (0.7, 0.9, 1),
    "VH": (0.9, 1, 1)
}

# =====================================================
# FONCTIONS UTILES
# =====================================================

def convertir(valeur, dictionnaire, personnalise=False):
    """Transforme une note en nombre flou triangulaire."""

    if not personnalise:
        code = str(valeur).strip().upper()

        if code not in dictionnaire:
            raise ValueError(
                f"Note linguistique inconnue : {valeur}"
            )

        return np.array(dictionnaire[code], dtype=float)

    # Saisie sous forme : 3,5,7
    texte = str(valeur).strip()
    texte = texte.replace("(", "").replace(")", "")
    elements = texte.split(",")

    if len(elements) != 3:
        raise ValueError(
            f"Triplet incorrect : {valeur}. "
            "Utiliser le format a,b,c."
        )

    try:
        a, b, c = [float(x.strip()) for x in elements]
    except ValueError:
        raise ValueError(
            f"Valeurs non numeriques : {valeur}"
        )

    if not np.all(np.isfinite([a, b, c])):
        raise ValueError("Les valeurs doivent etre finies.")

    if not (0 <= a <= b <= c):
        raise ValueError(
            f"Triplet invalide : {valeur}. "
            "Il faut 0 <= a <= b <= c."
        )

    return np.array([a, b, c], dtype=float)


def afficher_triplet(t):
    """Format lisible pour les matrices."""
    return f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"


def tableau_flou(matrice, lignes, colonnes):
    """Convertit une matrice floue en DataFrame lisible."""

    resultat = []

    for i in range(len(lignes)):
        ligne = []

        for j in range(len(colonnes)):
            ligne.append(afficher_triplet(matrice[i, j]))

        resultat.append(ligne)

    return pd.DataFrame(
        resultat, index=lignes, columns=colonnes
    )


def calcul_topsis(X, P, types):
    """
    X : candidats x decideurs x criteres x 3
    P : decideurs x criteres x 3
    """

    # 1. Matrice collective
    D = np.mean(X, axis=1)

    # 2. Normalisation
    R = np.zeros_like(D)
    references = []

    for j, nature in enumerate(types):

        if nature == "Benefice":

            maximum = np.max(D[:, j, 2])

            if maximum <= 0:
                raise ValueError(
                    f"Critere {j+1} : maximum nul."
                )

            R[:, j, :] = D[:, j, :] / maximum
            references.append(maximum)

        else:
            minimum = np.min(D[:, j, 0])

            if minimum <= 0:
                raise ValueError(
                    f"Critere {j+1} : pour un critere cout, "
                    "les valeurs doivent etre strictement positives."
                )

            R[:, j, 0] = minimum / D[:, j, 2]
            R[:, j, 1] = minimum / D[:, j, 1]
            R[:, j, 2] = minimum / D[:, j, 0]

            references.append(minimum)

    # 3. Poids collectifs
    W = np.mean(P, axis=0)

    # 4. Matrice normalisee ponderee
    V = R * W[None, :, :]

    # 5. Solutions ideales fixes du cours
    ideal_plus = np.ones(3)
    ideal_moins = np.zeros(3)

    # Distance floue : vertex method
    distance_plus = np.sqrt(
        np.mean((V - ideal_plus)**2, axis=2)
    )

    distance_moins = np.sqrt(
        np.mean((V - ideal_moins)**2, axis=2)
    )

    D_plus = np.sum(distance_plus, axis=1)
    D_moins = np.sum(distance_moins, axis=1)

    # Coefficient de proximite
    denominateur = D_plus + D_moins

    if np.any(denominateur == 0):
        raise ValueError("Distances nulles : calcul impossible.")

    CC = D_moins / denominateur

    return {
        "D": D,
        "R": R,
        "W": W,
        "V": V,
        "references": references,
        "D_plus": D_plus,
        "D_moins": D_moins,
        "CC": CC,
        "distance_plus": distance_plus,
        "distance_moins": distance_moins
    }


# =====================================================
# DONNEES DE L'EXERCICE ORIGINAL
# =====================================================

evaluation_exemple = [
    [
        ["MG", "G", "F", "VG", "F"],
        ["G", "MG", "G", "G", "F"],
        ["MG", "F", "G", "VG", "F"]
    ],
    [
        ["G", "VG", "VG", "VG", "VG"],
        ["G", "VG", "VG", "VG", "MG"],
        ["MG", "VG", "G", "VG", "G"]
    ],
    [
        ["VG", "MG", "G", "G", "G"],
        ["G", "G", "MG", "VG", "G"],
        ["F", "VG", "VG", "MG", "MG"]
    ]
]

importance_exemple = [
    ["H", "VH", "VH", "VH", "M"],
    ["VH", "VH", "H", "VH", "MH"],
    ["MH", "VH", "H", "VH", "MH"]
]

# =====================================================
# ETAT DE L'APPLICATION
# =====================================================

if "version" not in st.session_state:
    st.session_state.version = 0

if "exemple" not in st.session_state:
    st.session_state.exemple = True


def changer_probleme(mode):
    st.session_state.exemple = (mode == "Exemple du cours")
    st.session_state.version += 1


# =====================================================
# TITRE PRINCIPAL
# =====================================================

st.title("Fuzzy TOPSIS")

st.markdown(
    "### Systeme interactif d'aide a la decision multicritere"
)

st.write(
    "Configurez votre probleme, saisissez les evaluations "
    "et visualisez les etapes de la methode Fuzzy TOPSIS."
)

st.divider()

# =====================================================
# SIDEBAR : PARAMETRES
# =====================================================

with st.sidebar:

    st.header("Parametres")

    mode = st.radio(
        "Choisir le mode",
        ["Exemple du cours", "Nouveau probleme"],
        index=0 if st.session_state.exemple else 1,
        key=f"mode_{st.session_state.version}"
    )

    if mode == "Exemple du cours" and not st.session_state.exemple:
        changer_probleme(mode)
        st.rerun()

    if mode == "Nouveau probleme" and st.session_state.exemple:
        changer_probleme(mode)
        st.rerun()

    if st.session_state.exemple:
        na, nc, nd = 3, 5, 3
        saisie = "Linguistique"

        st.info(
            "Exemple initial : 3 candidats, "
            "5 criteres et 3 decideurs."
        )

    else:
        na = st.number_input(
            "Nombre d'alternatives",
            min_value=2,
            max_value=50,
            value=3
        )

        nc = st.number_input(
            "Nombre de criteres",
            min_value=1,
            max_value=30,
            value=3
        )

        nd = st.number_input(
            "Nombre de decideurs",
            min_value=1,
            max_value=20,
            value=2
        )

        saisie = st.radio(
            "Type de valeurs",
            ["Linguistique", "Nombres triangulaires"]
        )

    st.divider()

    st.subheader("Etapes de calcul")
    st.write("1. Matrice collective")
    st.write("2. Normalisation")
    st.write("3. Poids collectifs")
    st.write("4. Matrice ponderee")
    st.write("5. Classement final")

# =====================================================
# CONFIGURATION DU PROBLEME
# =====================================================

st.header("1. Configuration du probleme")

alternatives = [f"A{i+1}" for i in range(na)]
criteres = [f"C{j+1}" for j in range(nc)]
decideurs = [f"DM{k+1}" for k in range(nd)]

configuration = pd.DataFrame({
    "Critere": criteres,
    "Nature": ["Benefice"] * nc
})

if st.session_state.exemple:
    noms_defaut = [
        "Stabilite emotionnelle",
        "Communication orale",
        "Personnalite",
        "Experience passee",
        "Confiance en soi"
    ]
    configuration["Nom"] = noms_defaut
else:
    configuration["Nom"] = criteres

st.subheader("Definition des criteres")

configuration = st.data_editor(
    configuration,
    hide_index=True,
    use_container_width=True,
    disabled=["Critere"],
    column_config={
        "Nature": st.column_config.SelectboxColumn(
            "Nature",
            options=["Benefice", "Cout"],
            required=True
        ),
        "Nom": st.column_config.TextColumn("Nom")
    },
    key=f"config_{st.session_state.version}_{nc}"
)

types = configuration["Nature"].tolist()

with st.expander("Modifier les noms des alternatives"):

    df_noms = pd.DataFrame({
        "Identifiant": alternatives,
        "Nom": alternatives
    })

    df_noms = st.data_editor(
        df_noms,
        hide_index=True,
        disabled=["Identifiant"],
        key=f"noms_{st.session_state.version}_{na}"
    )

    alternatives = df_noms["Nom"].tolist()

# =====================================================
# SAISIE DES EVALUATIONS
# =====================================================

st.divider()
st.header("2. Saisie des evaluations")

st.write(
    "Chaque decideur evalue chaque alternative "
    "selon tous les criteres."
)

if saisie == "Linguistique":
    st.info(
        "Notes autorisees : VP, P, MP, F, MG, G, VG."
    )
else:
    st.info(
        "Saisissez chaque nombre flou au format : 3,5,7"
    )

lignes_eval = []

for i in range(na):
    for k in range(nd):

        ligne = {
            "Alternative": alternatives[i],
            "Decideur": decideurs[k]
        }

        for j in range(nc):

            if st.session_state.exemple:
                valeur = evaluation_exemple[i][k][j]
            else:
                valeur = (
                    "F" if saisie == "Linguistique"
                    else "3,5,7"
                )

            ligne[criteres[j]] = valeur

        lignes_eval.append(ligne)

df_eval = pd.DataFrame(lignes_eval)

st.subheader("Matrice des evaluations")

df_eval = st.data_editor(
    df_eval,
    hide_index=True,
    use_container_width=True,
    disabled=["Alternative", "Decideur"],
    key=(
        f"eval_{st.session_state.version}_"
        f"{na}_{nc}_{nd}_{saisie}"
    )
)

# =====================================================
# SAISIE DES POIDS
# =====================================================

st.subheader("Importance des criteres")

lignes_poids = []

for k in range(nd):

    ligne = {"Decideur": decideurs[k]}

    for j in range(nc):

        if st.session_state.exemple:
            valeur = importance_exemple[k][j]
        else:
            valeur = (
                "H" if saisie == "Linguistique"
                else "0.5,0.7,0.9"
            )

        ligne[criteres[j]] = valeur

    lignes_poids.append(ligne)

df_poids = pd.DataFrame(lignes_poids)

df_poids = st.data_editor(
    df_poids,
    hide_index=True,
    use_container_width=True,
    disabled=["Decideur"],
    key=(
        f"poids_{st.session_state.version}_"
        f"{nc}_{nd}_{saisie}"
    )
)

# =====================================================
# AFFICHAGE DES ECHELLES
# =====================================================

with st.expander("Consulter les echelles linguistiques"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Evaluations**")
        st.dataframe(
            pd.DataFrame(
                [(k, str(v)) for k, v in notes.items()],
                columns=["Code", "Nombre flou"]
            ),
            hide_index=True
        )

    with col2:
        st.markdown("**Poids**")
        st.dataframe(
            pd.DataFrame(
                [(k, str(v)) for k, v in poids.items()],
                columns=["Code", "Nombre flou"]
            ),
            hide_index=True
        )

# =====================================================
# CONVERSION DES DONNEES
# =====================================================

personnalise = saisie == "Nombres triangulaires"

try:

    X = np.zeros((na, nd, nc, 3))
    P = np.zeros((nd, nc, 3))

    # Evaluations
    for i in range(na):
        for k in range(nd):
            indice = i * nd + k

            for j in range(nc):
                X[i, k, j] = convertir(
                    df_eval.iloc[indice][criteres[j]],
                    notes,
                    personnalise
                )

    # Poids
    for k in range(nd):
        for j in range(nc):

            P[k, j] = convertir(
                df_poids.iloc[k][criteres[j]],
                poids,
                personnalise
            )

            if np.any(P[k, j] > 1):
                raise ValueError(
                    "Les poids doivent appartenir a [0,1]."
                )

    resultat = calcul_topsis(X, P, types)

except (ValueError, TypeError, KeyError) as erreur:

    st.error(f"Erreur dans les donnees : {erreur}")
    st.stop()

# =====================================================
# RESULTATS DETAILLES
# =====================================================

st.divider()
st.header("3. Calculs Fuzzy TOPSIS")

D = resultat["D"]
R = resultat["R"]
W = resultat["W"]
V = resultat["V"]

# -----------------------------------------------------
# QUESTION 1
# -----------------------------------------------------

st.subheader("Etape 1 : Matrice collective")

st.latex(
    r"\tilde{x}_{ij}="
    r"\frac{1}{K}\sum_{k=1}^{K}\tilde{x}_{ij}^{(k)}"
)

st.write(
    "On calcule la moyenne des evaluations "
    "des differents decideurs."
)

st.dataframe(
    tableau_flou(D, alternatives, criteres),
    use_container_width=True
)

with st.expander("Voir un exemple de calcul : matrice collective"):

    a = st.selectbox(
        "Alternative",
        range(na),
        format_func=lambda i: alternatives[i],
        key="ex1a"
    )

    c = st.selectbox(
        "Critere",
        range(nc),
        format_func=lambda j: criteres[j],
        key="ex1c"
    )

    st.write("Evaluations des decideurs :")

    for k in range(nd):
        st.write(
            f"{decideurs[k]} : {afficher_triplet(X[a,k,c])}"
        )

    st.write(
        f"Moyenne des {nd} decideurs : "
        f"{afficher_triplet(D[a,c])}"
    )

# -----------------------------------------------------
# QUESTION 2
# -----------------------------------------------------

st.subheader("Etape 2 : Normalisation")

st.write("Pour un critere de benefice :")

st.latex(
    r"\tilde{r}_{ij}="
    r"\left("
    r"\frac{a_{ij}}{c_j^*},"
    r"\frac{b_{ij}}{c_j^*},"
    r"\frac{c_{ij}}{c_j^*}"
    r"\right)"
)

st.write("Pour un critere de cout :")

st.latex(
    r"\tilde{r}_{ij}="
    r"\left("
    r"\frac{a_j^-}{c_{ij}},"
    r"\frac{a_j^-}{b_{ij}},"
    r"\frac{a_j^-}{a_{ij}}"
    r"\right)"
)

st.dataframe(
    tableau_flou(R, alternatives, criteres),
    use_container_width=True
)

with st.expander("Voir un exemple de normalisation"):

    a2 = st.selectbox(
        "Alternative",
        range(na),
        format_func=lambda i: alternatives[i],
        key="ex2a"
    )

    c2 = st.selectbox(
        "Critere",
        range(nc),
        format_func=lambda j: criteres[j],
        key="ex2c"
    )

    st.write("Valeur initiale :", afficher_triplet(D[a2,c2]))

    st.write("Nature :", types[c2])

    st.write(
        "Reference de normalisation :",
        round(resultat["references"][c2], 4)
    )

    st.write(
        "Valeur normalisee :",
        afficher_triplet(R[a2,c2])
    )

# -----------------------------------------------------
# QUESTION 3
# -----------------------------------------------------

st.subheader("Etape 3 : Poids collectifs")

st.latex(
    r"\tilde{w}_j="
    r"\frac{1}{K}\sum_{k=1}^{K}\tilde{w}_j^{(k)}"
)

st.write(
    "Les poids collectifs correspondent a la "
    "moyenne des poids attribues par les decideurs."
)

df_W = pd.DataFrame({
    "Critere": criteres,
    "Nom": configuration["Nom"],
    "Poids collectif": [afficher_triplet(x) for x in W]
})

st.dataframe(
    df_W,
    hide_index=True,
    use_container_width=True
)

with st.expander("Voir un exemple de calcul des poids"):

    c3 = st.selectbox(
        "Critere",
        range(nc),
        format_func=lambda j: criteres[j],
        key="ex3c"
    )

    for k in range(nd):
        st.write(
            f"{decideurs[k]} : {afficher_triplet(P[k,c3])}"
        )

    st.write(
        "Poids collectif :",
        afficher_triplet(W[c3])
    )

# -----------------------------------------------------
# QUESTION 4
# -----------------------------------------------------

st.subheader("Etape 4 : Matrice normalisee ponderee")

st.latex(
    r"\tilde{v}_{ij}="
    r"\tilde{r}_{ij}\otimes\tilde{w}_j"
)

st.write(
    "On multiplie chaque evaluation normalisee "
    "par le poids collectif du critere correspondant."
)

st.dataframe(
    tableau_flou(V, alternatives, criteres),
    use_container_width=True
)

with st.expander("Voir un exemple de ponderation"):

    a4 = st.selectbox(
        "Alternative",
        range(na),
        format_func=lambda i: alternatives[i],
        key="ex4a"
    )

    c4 = st.selectbox(
        "Critere",
        range(nc),
        format_func=lambda j: criteres[j],
        key="ex4c"
    )

    st.write(
        "Valeur normalisee :",
        afficher_triplet(R[a4,c4])
    )

    st.write(
        "Poids du critere :",
        afficher_triplet(W[c4])
    )

    st.write(
        "Produit :",
        afficher_triplet(V[a4,c4])
    )

# -----------------------------------------------------
# QUESTION 5
# -----------------------------------------------------

st.subheader("Etape 5 : Classement des alternatives")

st.write("Solution ideale positive :")

st.latex(r"A^+=(1,1,1)")

st.write("Solution ideale negative :")

st.latex(r"A^-=(0,0,0)")

st.write("Distance entre deux nombres flous :")

st.latex(
    r"d(\tilde{a},\tilde{b})="
    r"\sqrt{\frac{1}{3}"
    r"\left[(a_1-b_1)^2+(a_2-b_2)^2+"
    r"(a_3-b_3)^2\right]}"
)

st.write("Coefficient de proximite :")

st.latex(
    r"CC_i=\frac{D_i^-}{D_i^++D_i^-}"
)

# Classement
classement = pd.DataFrame({
    "Alternative": alternatives,
    "D+": resultat["D_plus"],
    "D-": resultat["D_moins"],
    "CC": resultat["CC"]
})

classement = classement.sort_values(
    "CC", ascending=False
).reset_index(drop=True)

classement.insert(
    0, "Rang", range(1, len(classement)+1)
)

st.dataframe(
    classement.style.format({
        "D+": "{:.4f}",
        "D-": "{:.4f}",
        "CC": "{:.4f}"
    }),
    hide_index=True,
    use_container_width=True
)

# =====================================================
# VISUALISATION
# =====================================================

st.subheader("Comparaison graphique")

graphique = classement.set_index("Alternative")["CC"]

st.bar_chart(
    graphique,
    color="#A64270",
    horizontal=True
)

# =====================================================
# MEILLEURE ALTERNATIVE
# =====================================================

meilleur = classement.iloc[0]

st.success(
    f"Meilleure alternative : {meilleur['Alternative']} "
    f"avec un coefficient de proximite "
    f"de {meilleur['CC']:.4f}"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Meilleure alternative",
        meilleur["Alternative"]
    )

with col2:
    st.metric(
        "Coefficient CC",
        f"{meilleur['CC']:.4f}"
    )

with col3:
    st.metric(
        "Alternatives analysees",
        na
    )

# =====================================================
# DETAIL DES DISTANCES
# =====================================================

with st.expander("Detail des distances par critere"):

    selection = st.selectbox(
        "Choisir une alternative",
        range(na),
        format_func=lambda i: alternatives[i]
    )

    df_distances = pd.DataFrame({
        "Critere": criteres,
        "Distance positive":
            resultat["distance_plus"][selection],
        "Distance negative":
            resultat["distance_moins"][selection]
    })

    st.dataframe(
        df_distances,
        hide_index=True,
        use_container_width=True
    )

    st.write(
        "Somme des distances positives :",
        round(resultat["D_plus"][selection], 4)
    )

    st.write(
        "Somme des distances negatives :",
        round(resultat["D_moins"][selection], 4)
    )

    st.write(
        "Coefficient final :",
        round(resultat["CC"][selection], 4)
    )

# =====================================================
# EXPORT DES RESULTATS
# =====================================================

st.divider()
st.header("4. Exportation des resultats")

csv = classement.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Exporter le classement en CSV",
    data=csv,
    file_name="classement_fuzzy_topsis.csv",
    mime="text/csv"
)

st.caption(
    "Fuzzy TOPSIS - Analyse multicritere avec "
    "nombres flous triangulaires."
)
