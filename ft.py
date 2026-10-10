
import numpy as np

# ETAPE 1 : Variables linguistiques

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

# Tableau 2 : evaluations des candidats
# Chaque candidat contient les avis de DM1, DM2, DM3

evaluations = [
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

# Tableau 3 : importance des criteres

importance = [
    ["H", "VH", "VH", "VH", "M"],
    ["VH", "VH", "H", "VH", "MH"],
    ["MH", "VH", "H", "VH", "MH"]
]

# Conversion en nombres flous

X = np.array([
    [[notes[n] for n in dm] for dm in candidat]
    for candidat in evaluations
], dtype=float)

# QUESTION 1 : Matrice collective
D = X.mean(axis=1)

# QUESTION 2 : Normalisation
maximums = D[:, :, 2].max(axis=0)
R = D / maximums[None, :, None]

# QUESTION 3 : Poids collectifs
W = np.array([
    [poids[n] for n in dm]
    for dm in importance
], dtype=float).mean(axis=0)

# QUESTION 4 : Matrice ponderee
V = R * W[None, :, :]

# QUESTION 5 : Classement TOPSIS

# Distance a l'ideal positif (1,1,1)
D_plus = np.sqrt(
    np.mean((V - 1)**2, axis=2)
).sum(axis=1)

# Distance a l'ideal negatif (0,0,0)
D_moins = np.sqrt(
    np.mean(V**2, axis=2)
).sum(axis=1)

# Coefficients de proximite
CC = D_moins / (D_plus + D_moins)

# Affichage
np.set_printoptions(precision=4, suppress=True)

print("1. Matrice collective :\n", D)
print("2. Matrice normalisee :\n", R)
print("3. Poids collectifs :\n", W)
print("4. Matrice ponderee :\n", V)

print("\n5. Resultats finaux")
for i in range(3):
    print(f"A{i+1} : CC = {CC[i]:.4f}")

classement = np.argsort(-CC)
print("Classement :",
      " > ".join(f"A{i+1}" for i in classement))
