# Importer des paquets
from __future__ import print_function
import cv2
import os

# On définit les paramètres (nom de la vidéo)
filename = 'GOPR10.mp4'
file_input = os.path.join(os.getcwd(), 'video', filename)
file_output = os.path.join(os.getcwd(), 'output', filename[:-4] + '.avi')


# On définit la capture pour la vidéo entrante: cap
cap = cv2.VideoCapture(file_input)

# Eliminer le fichier (si il existe)
if os.path.isfile(file_output):
    os.remove(file_output)

# On définit la vidéo d'entrée
background_extractor = cv2.createBackgroundSubtractorMOG2()

# Paramètres de la vidéo de sortie
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(file_output, fourcc, fps, (int(width), int(height)))


# On désactive OpenCL, celui-ci sert seulement si l'ordinateur possède une carte graphique,
# si on ne fait pas cela la suite ne pourra pas fonctionner sur des ordinateurs qui ne la possèdent pas
cv2.ocl.setUseOpenCL(False)

cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

# Si le fichier est ouvert, continuer
while cap.isOpened():

    # Lire la frame
    ret, frame = cap.read()

    # Si la frame existe,...
    if ret:

        # Grâce à cela on peut savoir si une frame est en procès et si oui laquelle
        print('Frame %d en procès' % cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Extraction de fond
        fgmask = background_extractor.apply(frame)

        # Extraction de contours pour identifier les objets mobiles
        im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Pour chaque contour faire:
        for contour in contours:

            # Si le contour a une aire supérieure à 30 pixels carrés
            # et inférieure 70 pixels carrés, continuer
            if cv2.contourArea(contour) >30 and cv2.contourArea(contour) <70:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Redimmensioner la vidéo
        frame_resized = cv2.resize(frame, (960, 500))
        fgmask_resized = cv2.resize(fgmask, (960, 500))

        # Visualiser la vidéo
        cv2.imshow('Camera', frame_resized)
        cv2.imshow('Seuil', fgmask_resized)

        # Convertir la frame procéssé à RGB et l'ecrire dans la vidéo de sortie,
        # car le format ".avi" ne peut pas supporter d'autres espaces colirimétriques
        frame_vid = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

        # Écrire la frame dans le fichier
        out.write(frame_vid)

    # ...mais si la frame n'existe pas, sortir
    else:
        break

        # On peut sortir en appuyant la touche "s"
    k = cv2.waitKey(30) & 0xff
    if k == ord("s"):
        break

# Libérer le fichier d'entrée et celui en sortie et fermer les fenêtres
cap.release()
out.release()
cv2.destroyAllWindows()
