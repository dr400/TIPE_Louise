import cv2
import numpy as np
import json

# ──────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────
CALIB_FILE     = "stereo_calibration_data.npz"  # produit par Calibration.py
GOAL_LINE_FILE = "goal_line.json"               # fichier de sortie
CAM_GAUCHE     = 0
CAM_DROITE     = 1


# ──────────────────────────────────────────────
# CHARGEMENT DE LA CALIBRATION
# ──────────────────────────────────────────────
def charger_calibration(chemin: str) -> dict:
    """Charge les paramètres stéréo et construit les matrices de projection."""
    data = np.load(chemin)
    mtx_g  = data["mtx_l"].astype(np.float64)
    dist_g = data["dist_l"].astype(np.float64)
    mtx_d  = data["mtx_r"].astype(np.float64)
    dist_d = data["dist_r"].astype(np.float64)
    R      = data["R"].astype(np.float64)
    T      = data["T"].astype(np.float64)

    # Caméra gauche = repère de référence
    P1 = mtx_g @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = mtx_d @ np.hstack([R, T])

    return {
        "mtx_g": mtx_g, "dist_g": dist_g,
        "mtx_d": mtx_d, "dist_d": dist_d,
        "R": R, "T": T,
        "P1": P1, "P2": P2,
    }


# ──────────────────────────────────────────────
# SÉLECTION INTERACTIVE DE POINTS
# ──────────────────────────────────────────────
_points_cliques: list = []

def _clic_souris(evenement, x, y, flags, param):
    if evenement == cv2.EVENT_LBUTTONDOWN and len(_points_cliques) < 2:
        _points_cliques.append((x, y))


def selectionner_deux_points(frame: np.ndarray, titre: str):
    """Affiche `frame`, demande à l'utilisateur de cliquer 2 points.
    Retourne (pt1, pt2) en pixels."""
    global _points_cliques
    _points_cliques = []
    cv2.namedWindow(titre)
    cv2.setMouseCallback(titre, _clic_souris)

    while True:
        affichage = frame.copy()
        for pt in _points_cliques:
            cv2.circle(affichage, pt, 6, (0, 255, 0), -1)
        if len(_points_cliques) == 2:
            cv2.line(affichage, _points_cliques[0], _points_cliques[1], (0, 0, 255), 2)
            cv2.putText(
                affichage,
                "OK ? [Q] valider  [R] recommencer",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
        cv2.imshow(titre, affichage)
        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q") and len(_points_cliques) == 2:
            break
        if touche == ord("r"):
            _points_cliques = []

    cv2.destroyAllWindows()
    return tuple(_points_cliques[0]), tuple(_points_cliques[1])


# ──────────────────────────────────────────────
# TRIANGULATION
# ──────────────────────────────────────────────
def corriger_distorsion(pt, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    pts = np.array([[pt]], dtype=np.float64)
    return cv2.undistortPoints(pts, K, D, P=K)[0][0]


def trianguler(pt_gauche, pt_droite, calib: dict) -> np.ndarray:
    """Retourne les coordonnées 3D (repère caméra gauche) d'un point."""
    u_g = corriger_distorsion(pt_gauche, calib["mtx_g"], calib["dist_g"])
    u_d = corriger_distorsion(pt_droite, calib["mtx_d"], calib["dist_d"])
    pts4 = cv2.triangulatePoints(
        calib["P1"], calib["P2"],
        u_g.reshape(2, 1), u_d.reshape(2, 1),
    )
    return (pts4[:3] / pts4[3]).flatten()


# ──────────────────────────────────────────────
# REPROJECTION (vérification visuelle)
# ──────────────────────────────────────────────
def reprojeter_ligne(
    frame: np.ndarray,
    A_3d: np.ndarray,
    B_3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray = None,
    T: np.ndarray = None,
) -> None:
    """Dessine la ligne de but reprojetée sur `frame`."""
    R_use = R if R is not None else np.eye(3)
    T_use = T if T is not None else np.zeros((3, 1))
    rvec, _ = cv2.Rodrigues(R_use)

    def projeter(pt3d):
        p2d, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, T_use, K, D)
        return tuple(p2d[0][0].astype(int))

    pA, pB = projeter(A_3d), projeter(B_3d)
    cv2.line(frame, pA, pB, (0, 0, 255), 3)
    cv2.circle(frame, pA, 6, (0, 255, 0), -1)
    cv2.circle(frame, pB, 6, (0, 255, 0), -1)


# ──────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ──────────────────────────────────────────────
def main():
    # 1. Calibration
    calib = charger_calibration(CALIB_FILE)

    # 2. Capture d'une image depuis chaque caméra
    cap_g = cv2.VideoCapture(CAM_GAUCHE)
    cap_d = cv2.VideoCapture(CAM_DROITE)
    ok_g, frame_g = cap_g.read()
    ok_d, frame_d = cap_d.read()
    cap_g.release()
    cap_d.release()
    if not ok_g or not ok_d:
        raise RuntimeError("Impossible de lire une image depuis l'une des caméras.")

    # 3. Sélection des deux extrémités de la ligne de but (caméra gauche)
    ptA_g, ptB_g = selectionner_deux_points(frame_g, "Caméra Gauche — ligne de but")

    # 4. Mêmes points sur la caméra droite
    ptA_d, ptB_d = selectionner_deux_points(frame_d, "Caméra Droite — ligne de but")

    # 5. Triangulation → coordonnées 3D
    A_3d = trianguler(ptA_g, ptA_d, calib)
    B_3d = trianguler(ptB_g, ptB_d, calib)
    print(f"A_3d = {A_3d}")
    print(f"B_3d = {B_3d}")

    # 6. Sauvegarde dans goal_line.json
    ligne = {"A": A_3d.tolist(), "B": B_3d.tolist()}
    with open(GOAL_LINE_FILE, "w") as f:
        json.dump(ligne, f, indent=2)
    print(f"Ligne de but sauvegardée dans '{GOAL_LINE_FILE}'.")

    # 7. Vérification visuelle
    print("Vérification visuelle — appuyez sur une touche pour fermer.")
    reprojeter_ligne(frame_g, A_3d, B_3d, calib["mtx_g"], calib["dist_g"])
    reprojeter_ligne(
        frame_d, A_3d, B_3d,
        calib["mtx_d"], calib["dist_d"],
        calib["R"], calib["T"],
    )
    cv2.imshow("Caméra Gauche — Vérification", frame_g)
    cv2.imshow("Caméra Droite — Vérification", frame_d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
