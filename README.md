# Atlantic Haven Hotels - Prédiction d'annulation de réservation
# Hackaton ML & M1 ISPM
##  En-tête institutionnel et Identification

### 1. Lien hypertexte vers le site officiel de l'institut :
https://ispm-edu.com

### 2. Nom du groupe de projet : **Lemurien-Codeur**


### 3. Tableau listant les membres de l'équipe :

| 		**Nom Complet**                       | **Numéro d'étudiant** |  **Classe** |
|-------------------------------------------------|-------------------|---------|
| RANAIVONOHATRA Mahenintsoa Akel                   | 	   02	      | ESIIA 4 |
| RAJOELISOLO Sitraka Tsitohaina                  | 	   05	      | ESIIA 4 |
| ANDRISOAMALALA Volakanto Landréa                | 	   10	      | ESIIA 4 |
| RAMESON Andrianarinosy Imanoela Fiderana Ny Avo | 	   12	      | ESIIA 4 |
|Razafimanantsoa Betina					  |	   23		| ESIIA 4 |
| TANG Fakanah Randy                              | 	   27	      | ESIIA 4 | 		                |
| LEONARD Jamaviston Lucas                        | 	   36	      | ESIIA 4 |

## Objectif
Prédire `reservation_annulee` (0 = maintenue, 1 = annulée), fournir une probabilité et une décision binaire, et maximiser le **F1-score** sur la classe « annulation », tout en respectant l'ordre temporel des données.

## Etape 1: EDA et préparation
Cette première étape permet de comprendre les données et de les préparer avant l'entraînement du modèle.

### 1.1 — Configuration de l'environnement

Importation des principales bibliothèques utilisées :

NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

Ces bibliothèques permettent de manipuler les données, réaliser les visualisations et construire les modèles de Machine Learning.

### 1.2 — Chargement des données

Les différents fichiers CSV sont chargés :

reservations_train.csv → données utilisées pour l'apprentissage et la validation.
reservations_test.csv → données utilisées pour les prédictions finales.
Le dictionnaire des données → permet de comprendre la signification des variables.

### 1.3 — Inspection du dictionnaire des variables

Le dictionnaire des données est consulté afin d'identifier :

le nom des variables ;
leur signification ;
leur type ;
leur rôle dans le problème de prédiction.

Cette étape permet de mieux comprendre le jeu de données avant de procéder au nettoyage.

### 1.4 — Nettoyage et harmonisation des types

Les types des différentes colonnes sont corrigés et harmonisés.

Les principales opérations concernent notamment :

la conversion des variables de type date ;
la conversion des variables binaires ;
la standardisation des types nécessaires aux étapes suivantes.

### 1.5 — Audit des valeurs manquantes

Un bilan des valeurs manquantes est réalisé sur les données d'entraînement.

Cette analyse permet d'identifier les colonnes contenant des NaN et de déterminer quelles variables nécessitent un traitement particulier.

### 1.6 — Imputation métier des valeurs manquantes

Les valeurs manquantes sont remplacées en utilisant une logique adaptée à la signification des variables.

L'objectif est de ne pas simplement remplacer toutes les valeurs manquantes par une valeur arbitraire, mais de tenir compte du contexte métier des réservations hôtelières.

### 1.7 — Validation du nettoyage

Une nouvelle vérification est effectuée après le traitement.

L'objectif est notamment de vérifier qu'il ne reste plus de valeurs manquantes dans les jeux de données nettoyés.

## Etape 2: Baseline Obligatoire
### 2.1 — Analyse visuelle des données

Des visualisations sont réalisées afin de mieux comprendre les données et d'identifier les éventuelles relations entre les variables et l'annulation des réservations.

Cette étape permet notamment d'obtenir une première compréhension du comportement de la variable cible.

### 2.2 — Split temporel

Les données sont séparées en fonction du temps plutôt qu'avec un découpage aléatoire classique.

Cette méthode est importante car le problème possède une dimension temporelle.

Le principe est de :

**Données passées → entraînement**

**Données plus récentes → validation**

Cela permet de reproduire davantage une situation réelle de prédiction future et de limiter les risques de fuite de données.

## Etape 3: Validation et modelisation
Un **ColumnTransformer** est construit afin d'appliquer automatiquement les traitements nécessaires aux différentes catégories de variables.

Le pipeline permet notamment de gérer :

-les variables numériques ;
-les variables catégorielles ;
-l'encodage des variables ;
-la préparation des données avant l'entraînement.

L'intérêt principal est de **regrouper les transformations dans une seule chaîne de traitement**.

## Etape 4: Featuring Engineering
Une **Régression Logistique** est utilisée comme modèle de référence (baseline).

Le modèle est intégré directement dans un pipeline avec le prétraitement.

La chaîne générale devient donc :

**Données → Prétraitement → Régression Logistique → Prédiction**

Le modèle produit une prédiction de la classe ainsi qu'une probabilité d'annulation.

## Etape 5: Interpretation et erreur
Le modèle est évalué sur les données de validation.

La principale métrique utilisée est le :

### F1-score

Le F1-score est particulièrement important ici car l'objectif est d'obtenir de bonnes performances sur la classe **"annulation"**.

Le **classification_report** permet également d'observer différentes métriques de classification, notamment :

-Precision
-Recall
-F1-score
-Support

Cette étape permet d'obtenir une première mesure des performances du modèle.

## Etape 6: Soumission
Une optimisation de la Régression Logistique est réalisée avec :

-GridSearchCV
-TimeSeriesSplit
-GridSearchCV

### GridSearchCV:
 teste différentes combinaisons d'hyperparamètres afin de rechercher la configuration offrant les meilleures performances.

### TimeSeriesSplit:

TimeSeriesSplit est utilisé pour conserver l'ordre chronologique des observations pendant la validation croisée.

Cela évite d'utiliser des données futures pour entraîner le modèle sur une période antérieure.

## ÉTAPE 7 — Recherche d'améliorations

Le notebook termine en proposant plusieurs pistes pour améliorer le F1-score obtenu avec la Régression Logistique.

Parmi les possibilités proposées :

### Random Forest:

Le RandomForestClassifier pourrait permettre de mieux capturer :

les relations non linéaires ;
les interactions entre variables.

Les principaux hyperparamètres pouvant être optimisés sont notamment :

-n_estimators
-max_depth
-min_samples_split
-min_samples_leaf

### Gradient Boosting

Les modèles de type Gradient Boosting Machine (GBM) constituent également une piste envisagée pour améliorer les performances par rapport à la baseline.

## Lien vidéo:
--> https://drive.google.com/file/d/1ES2QsNnhZlTY_h8pVdISUYx_Un7HvD_r/view?usp=drivesdk