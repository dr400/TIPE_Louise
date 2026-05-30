import numpy as np
import cv2
import glob
import os

# ──────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────
CHESSBOARD_SIZE  = (8, 4)   # coins intérieurs (colonnes-1, lignes-1)
SQUARE_SIZE_MM   = 30.0     # taille réelle d'une case en mm
CALIB_FILE       = "stereo_calibration_data.npz"
IMAGES_GAUCHE    = "images/gauche/*.png"
IMAGES_DROITE    = "images/droite/*.png"

# critère d'arrêt pour cornerSubPix et stereoCalibrate
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def construire_objpoints():
    """Retourne la grille 3-D du damier (z = 0)."""
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    return objp * SQUARE_SIZE_MM


def charger_paires_images():
    """Retourne les listes triées des chemins d'images gauche et droite."""
    gauche = sorted(glob.glob(IMAGES_GAUCHE))
    droite = sorted(glob.glob(IMAGES_DROITE))
    if len(gauche) != len(droite) or len(gauche) == 0:
        raise ValueError(
            f"Nombre d'images incompatible ou dossier vide : "
            f"{len(gauche)} gauche / {len(droite)} droite."
        )
    return gauche, droite


def detecter_coins(images_gauche, images_droite, objp):
    """Détecte les coins du damier sur chaque paire et retourne
    objpoints, imgpoints_gauche, imgpoints_droite, image_size."""
    objpoints       = []
    imgpoints_left  = []
    imgpoints_right = []
    image_size      = None
    paires_valides  = 0

    for path_g, path_d in zip(images_gauche, images_droite):
        img_g = cv2.imread(path_g)
        img_d = cv2.imread(path_d)
        if img_g is None or img_d is None:
            print(f"[AVERTISSEMENT] Impossible de lire : {path_g} ou {path_d}")
            continue

        gray_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_g.shape[1], gray_g.shape[0])  # (largeur, hauteur)

        ret_g, corn_g = cv2.findChessboardCorners(gray_g, CHESSBOARD_SIZE, None)
        ret_d, corn_d = cv2.findChessboardCorners(gray_d, CHESSBOARD_SIZE, None)

        if ret_g and ret_d:
            objpoints.append(objp)
            corn_g = cv2.cornerSubPix(gray_g, corn_g, (11, 11), (-1, -1), CRITERIA)
            corn_d = cv2.cornerSubPix(gray_d, corn_d, (11, 11), (-1, -1), CRITERIA)
            imgpoints_left.append(corn_g)
            imgpoints_right.append(corn_d)
            paires_valides += 1
        else:
            print(f"[INFO] Damier non détecté dans la paire : {os.path.basename(path_g)}")

    print(f"{paires_valides} paires valides sur {len(images_gauche)}.")
    if paires_valides < 5:
        raise RuntimeError("Pas assez de paires valides pour calibrer (minimum 5).")
    return objpoints, imgpoints_left, imgpoints_right, image_size


def calibrer_stereo():
    objp = construire_objpoints()
    images_g, images_d = charger_paires_images()
    objpoints, imgpoints_g, imgpoints_d, image_size = detecter_coins(images_g, images_d, objp)

    # ── Calibration intrinsèque individuelle ──
    print("Calibration caméra gauche…")
    ret_g, mtx_g, dist_g, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_g, image_size, None, None
    )
    print(f"  Erreur de reprojection gauche : {ret_g:.4f} px")

    print("Calibration caméra droite…")
    ret_d, mtx_d, dist_d, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_d, image_size, None, None
    )
    print(f"  Erreur de reprojection droite : {ret_d:.4f} px")

    # ── Calibration stéréo (extrinsèque) ──
    print("Calibration stéréo…")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_g, imgpoints_d,
        mtx_g, dist_g,
        mtx_d, dist_d,
        image_size,
        criteria=CRITERIA,
        flags=flags,
    )
    print(f"  Erreur stéréo : {ret_stereo:.4f} px")

    # ── Sauvegarde ──
    np.savez(
        CALIB_FILE,
        mtx_l=mtx_g, dist_l=dist_g,
        mtx_r=mtx_d, dist_r=dist_d,
        R=R, T=T,
        image_size=np.array(image_size),
    )
    print(f"Calibration sauvegardée dans '{CALIB_FILE}'.")
    return ret_stereo


if __name__ == "__main__":
    calibrer_stereo()
