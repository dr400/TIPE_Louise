# Calibration stéréo de caméras (damier)

Ce dépôt contient un notebook Jupyter qui calibre deux caméras à partir de paires d’images d’un damier.  
Le script calcule les paramètres intrinsèques de chaque caméra puis la géométrie relative entre elles, et sauvegarde les résultats dans un fichier `.npz`.

## Prérequis

- Python 3
- Bibliothèques : `numpy`, `opencv-python`

Installation :

```bash
pip install numpy opencv-python
```

## Structure attendue

Placez les images dans les dossiers suivants (même nombre d’images à gauche et à droite) :

```
images/
  gauche/
    *.jpg
  droite/
    *.jpg
```

## Utilisation

1. Ouvrir le notebook `Programme Calibration Caméras-2.ipynb`.
2. Exécuter la cellule principale.

Le programme :
- détecte les coins du damier sur chaque paire d’images,
- calcule les matrices intrinsèques et la distorsion de chaque caméra,
- calcule la rotation/translation entre les deux caméras,
- enregistre les résultats.

## Paramètres importants

Dans la fonction `calibrate_cameras()` :

- `chessboard_size = (8, 6)` : nombre de coins internes (colonnes, lignes).
- `square_size = 30.0` : taille d’une case du damier en millimètres.

Adaptez ces valeurs à votre damier.

## Sorties

Un fichier est généré à la racine :

```
stereo_calibration_data.npz
```

Il contient :
- `mtx_l`, `dist_l` : paramètres intrinsèques et distorsion (caméra gauche)
- `mtx_r`, `dist_r` : paramètres intrinsèques et distorsion (caméra droite)
- `R`, `T` : rotation et translation entre les deux caméras

## Remarques

- Les deux caméras doivent voir le damier sur chaque paire d’images.
- Les images sont lues au format `.jpg`. Adaptez l’extension si nécessaire.
