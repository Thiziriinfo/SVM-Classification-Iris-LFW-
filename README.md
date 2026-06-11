#  SVM Classification — Iris & Face Recognition (LFW)

**Exploration des machines à vecteurs de support sur deux cas d'usage : classification de données structurées et reconnaissance faciale sur images réelles**

---

## Objectifs du projet

Ce projet explore les SVMs sous deux angles complémentaires :

1. **Classification binaire sur données structurées** — dataset Iris, comparaison noyau linéaire vs polynomial, tuning par GridSearchCV
2. **Reconnaissance faciale** — classification binaire sur le dataset LFW (Labeled Faces in the Wild), impact des variables de bruit, réduction de dimension par PCA

L'idée centrale : comprendre comment le choix du noyau, la régularisation, et la dimensionnalité impactent les performances d'un SVM — et mesurer ces effets de façon rigoureuse.

---

## Stack technique

`Python` · `scikit-learn` · `NumPy` · `Matplotlib` · `PCA` · `GridSearchCV`

---

## Dataset

### Iris (classification structurée)
- Source : `sklearn.datasets.load_iris`
- Périmètre : classification binaire (classes 1 et 2), 2 features pour visualisation
- Normalisation : StandardScaler avant entraînement

### LFW — Labeled Faces in the Wild (reconnaissance faciale)
- Source : `sklearn.datasets.fetch_lfw_people`
- Périmètre : classification binaire entre deux personnalités publiques (Tony Blair vs Colin Powell)
- Features : luminosité moyenne par pixel (réduction du RGB en grayscale)
- Normalisation : centrage-réduction feature par feature

---

## Démarche

### 1. Classification Iris — noyaux linéaire vs polynomial

Comparaison des deux noyaux sur le même split train/test (50/50) avec GridSearchCV 5-folds :

- **Noyau linéaire** : recherche sur C ∈ [10⁻³, 10³] (200 valeurs log-space)
- **Noyau polynomial** : recherche sur C, gamma, degree (1, 2, 3)

Visualisation des frontières de décision pour interpréter les différences de comportement entre les deux noyaux.

### 2. Reconnaissance faciale — impact de la régularisation

Évaluation de l'accuracy sur X_test pour C ∈ [10⁻⁵, 10⁵] avec noyau linéaire. Sélection du meilleur C, visualisation des prédictions sur images test, et inspection des coefficients du modèle (heatmap des poids par pixel).

### 3. Robustesse aux variables de bruit

Expérience contrôlée : comparaison de la performance SVM avec et sans ajout de 300 features aléatoires (bruit gaussien σ=1).

| Configuration | Train score | Test score |
|--------------|-------------|------------|
| Sans bruit | ~0.97 | ~0.87 |
| Avec 300 features bruit | dégradation visible | dégradation visible |

### 4. Réduction de dimension par PCA

Application d'une PCA à 100 composantes sur les données bruitées avant entraînement SVM. Résultat : les performances retrouvent un niveau comparable au cas sans bruit, ce qui illustre concrètement l'utilité de la PCA comme étape de préprocessing en haute dimension.

---

## Résultats clés

- Le **noyau polynomial** surpasse légèrement le linéaire sur Iris grâce à sa capacité à modéliser des frontières non-linéaires
- Le **meilleur C** pour la reconnaissance faciale se situe entre 10⁻² et 10¹ — au-delà, le modèle surajuste
- **+300 features de bruit** dégradent les performances de façon mesurable : confirmation que la dimensionnalité est un facteur de risque
- **PCA à 100 composantes** restaure les performances en éliminant les dimensions non-informatives

---

## Structure du projet

```
├── src/
│   ├── code_source.py      # Pipeline principal : Iris + LFW + bruit + PCA
│   └── svm_source.py       # Fonctions utilitaires (plot_gallery, title, frontiere)
├── images/                 # Visualisations générées
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

---

## Lancer le projet

```bash
git clone https://github.com/Thiziriinfo/SVM-Classification-Iris-LFW.git
cd SVM-Classification-Iris-LFW
pip install scikit-learn numpy matplotlib
python src/code_source.py
```

> Le dataset LFW est téléchargé automatiquement par scikit-learn au premier lancement (~200MB).

---

## Ce que ce projet illustre

Les SVMs ont une réputation de "boîte noire" injustifiée — les coefficients du noyau linéaire sur LFW sont directement interprétables en heatmap sur les pixels. La visualisation des frontières de décision sur Iris montre immédiatement pourquoi un noyau polynomial peut être nécessaire sur des données non-linéairement séparables.

L'expérience bruit + PCA est la partie la plus instructive : elle reproduit un scénario réel où un dataset contient beaucoup de features non-informatives (ce qui est la règle, pas l'exception en pratique).

---

