"""Detection Balle.py

Détecte en temps réel une balle orange dans le flux stéréo et décide si elle
a franchi la ligne de but définie dans goal_line.json.

Logique :
  - La ligne de but est définie par deux points 3D A et B (repère caméra gauche).
  - Un plan de but est construit à partir de AB et de l'axe Z de la caméra.
  - Pour chaque frame, la balle orange est détectée par seuillage HSV dans
    les deux caméras, puis triangulée en 3D.
  - Si la balle est "derrière" la ligne (côté négatif du plan), BUT est affiché.

Pré-requis :
  - stereo_calibration_data.npz  (produit par Calibration.py)
  - goal_line.json               (produit par Position Ligne.py)
"""

import cv2
import numpy as np
import json

# ──────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────
CALIB_FILE     = "stereo_calibration_data.npz"
GOAL_LINE_FILE = "goal_line.json"
CAM_GAUCHE     = 0
CAM_DROITE     = 1

# Plage HSV pour la couleur orange.
# Ajuster si la détection est imprécise dans vos conditions d'éclairage.
ORANGE_BAS  = np.array([  5, 150,  80], dtype=np.uint8)
ORANGE_HAUT = np.array([ 25, 255, 255], dtype=np.uint8)

# Surface minimale du contour pour éviter les faux positifs (pixels²)
AIRE_MIN_BALLE = 300


# ──────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────
def charger_calibration(chemin: str) -> dict:
    data   = np.load(chemin)
    mtx_g  = data["mtx_l"].astype(np.float64)
    dist_g = data["dist_l"].astype(np.float64)
    mtx_d  = data["mtx_r"].astype(np.float64)
    dist_d = data["dist_r"].astype(np.float64)
    R      = data["R"].astype(np.float64)
    T      = data["T"].astype(np.float64)
    P1 = mtx_g @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = mtx_d @ np.hstack([R, T])
    return {
        "mtx_g": mtx_g, "dist_g": dist_g,
        "mtx_d": mtx_d, "dist_d": dist_d,
        "R": R, "T": T,
        "P1": P1, "P2": P2,
    }


def charger_ligne_but(chemin: str):
    """Retourne (A_3d, B_3d, normale, d) définissant le plan de but.

    Le plan est défini par : normale · X = d
    Un point X est DERRIÈRE la ligne si normale · X < d.
    """
    with open(chemin) as f:
        data = json.load(f)
    A = np.array(data["A"], dtype=np.float64)
    B = np.array(data["B"], dtype=np.float64)

    AB     = B - A
    axe_z  = np.array([0.0, 0.0, 1.0])          # la caméra regarde vers +Z
    normale = np.cross(AB, axe_z)
    norme   = np.linalg.norm(normale)
    if norme < 1e-9:
        raise ValueError(
            "La ligne de but est parallèle à l'axe optique — "
            "impossible de définir un plan de but."
        )
    normale /= norme
    d = float(np.dot(normale, A))
    return A, B, normale, d


# ──────────────────────────────────────────────
# DÉTECTION DE LA BALLE ORANGE
# ──────────────────────────────────────────────
def detecter_balle(frame: np.ndarray):
    """Retourne le centre (x, y) de la plus grande tache orange, ou None."""
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, ORANGE_BAS, ORANGE_HAUT)

    # Nettoyage morphologique pour supprimer le bruit
    noyau  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  noyau)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, masque

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < AIRE_MIN_BALLE:
        return None, masque

    ((cx, cy), _) = cv2.minEnclosingCircle(c)
    return (int(cx), int(cy)), masque


# ──────────────────────────────────────────────
# TRIANGULATION
# ──────────────────────────────────────────────
def corriger_distorsion(pt, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    pts = np.array([[pt]], dtype=np.float64)
    return cv2.undistortPoints(pts, K, D, P=K)[0][0]


def trianguler(pt_g, pt_d, calib: dict) -> np.ndarray:
    u_g  = corriger_distorsion(pt_g, calib["mtx_g"], calib["dist_g"])
    u_d  = corriger_distorsion(pt_d, calib["mtx_d"], calib["dist_d"])
    pts4 = cv2.triangulatePoints(
        calib["P1"], calib["P2"],
        u_g.reshape(2, 1), u_d.reshape(2, 1),
    )
    return (pts4[:3] / pts4[3]).flatten()


# ──────────────────────────────────────────────
# DÉCISION : balle derrière la ligne ?
# ──────────────────────────────────────────────
def est_derriere_ligne(balle_3d: np.ndarray, normale: np.ndarray, d: float) -> bool:
    """True si la balle se trouve du côté négatif du plan (dans les buts)."""
    return float(np.dot(normale, balle_3d)) < d


# ──────────────────────────────────────────────
# AFFICHAGE
# ──────────────────────────────────────────────
def dessiner_etat(
    frame: np.ndarray,
    centre,
    but: bool,
    balle_3d: np.ndarray = None,
) -> None:
    if centre is not None:
        couleur = (0, 0, 255) if but else (0, 255, 0)
        cv2.circle(frame, centre, 12, couleur, 3)
        if balle_3d is not None:
            cv2.putText(
                frame,
                f"Z={balle_3d[2]:.0f}mm",
                (centre[0] + 15, centre[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2,
            )
    if but:
        cv2.putText(frame, "BUT !", (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)


def reprojeter_ligne_but(
    frame: np.ndarray,
    A_3d: np.ndarray,
    B_3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray = None,
    T: np.ndarray = None,
) -> None:
    """Reporte la ligne de but en rouge sur l'image."""
    R_use = R if R is not None else np.eye(3)
    T_use = T if T is not None else np.zeros((3, 1))
    rvec, _ = cv2.Rodrigues(R_use)

    def proj(pt3d):
        p2d, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, T_use, K, D)
        return tuple(p2d[0][0].astype(int))

    cv2.line(frame, proj(A_3d), proj(B_3d), (0, 0, 220), 2)


# ──────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────
def main():
    calib            = charger_calibration(CALIB_FILE)
    A_3d, B_3d, normale, d = charger_ligne_but(GOAL_LINE_FILE)

    cap_g = cv2.VideoCapture(CAM_GAUCHE)
    cap_d = cv2.VideoCapture(CAM_DROITE)
    if not cap_g.isOpened() or not cap_d.isOpened():
        raise RuntimeError("Impossible d'ouvrir les caméras.")

    print("Détection en cours — [Q] pour quitter.")

    while True:
        ok_g, frame_g = cap_g.read()
        ok_d, frame_d = cap_d.read()
        if not ok_g or not ok_d:
            print("[ERREUR] Lecture caméra échouée.")
            break

        centre_g, _ = detecter_balle(frame_g)
        centre_d, _ = detecter_balle(frame_d)

        but      = False
        balle_3d = None

        if centre_g is not None and centre_d is not None:
            try:
                balle_3d = trianguler(centre_g, centre_d, calib)
                but      = est_derriere_ligne(balle_3d, normale, d)
            except Exception as e:
                print(f"[AVERTISSEMENT] Triangulation échouée : {e}")

        # Ligne de but reprojetée
        reprojeter_ligne_but(frame_g, A_3d, B_3d, calib["mtx_g"], calib["dist_g"])
        reprojeter_ligne_but(
            frame_d, A_3d, B_3d,
            calib["mtx_d"], calib["dist_d"],
            calib["R"], calib["T"],
        )

        dessiner_etat(frame_g, centre_g, but, balle_3d)
        dessiner_etat(frame_d, centre_d, but)

        cv2.imshow("Caméra Gauche", frame_g)
        cv2.imshow("Caméra Droite", frame_d)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_g.release()
    cap_d.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
