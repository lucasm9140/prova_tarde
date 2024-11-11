#TODO: Objetivo
 
  # 3. Distância Euclidiana (1D,2D,3D, n....) 
  # 4. Média (cálculo do EAR) 


import cv2 #pip install opencv-python
import mediapipe as mp #pip install mediapipe
import numpy as np
import time
import pygame
import os
import sys

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega o arquivo de som
pygame.mixer.music.load("brazil-eas-alarm-blood-moon-260034.mp3")

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))

    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.27
mar_limiar = 0.1
dormindo = 0
som_tocando = False
blinks_count = 0  # Contagem de piscadas
blink_time_threshold = 0.3  # Limite de tempo para considerar uma piscada (em segundos)
last_blink_time = 0  # Para controlar o tempo entre as piscadas

# capturar a câmera
cap = cv2.VideoCapture(0)

# desenhar os pontos
mp_drawing = mp.solutions.drawing_utils

# coletar solução do Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Função de controle de som
def controlar_som(mar, som_tocando):
    if mar >= mar_limiar:  # Se a boca estiver aberta
        if not som_tocando:  # Se o som não estiver tocando
            pygame.mixer.music.play(-1)  # Toca continuamente
            som_tocando = True
    else:  # Se a boca estiver fechada
        if som_tocando:  # Se o som estiver tocando
            pygame.mixer.music.stop()  # Para o som
            som_tocando = False
    return som_tocando

# Função para detectar piscadas e atualizar a contagem
def detectar_piscada(ear, last_blink_time, blinks_count):
    global blink_time_threshold
    if ear < ear_limiar:  # Se o EAR estiver abaixo do limiar, significa que o olho está fechado
        if time.time() - last_blink_time > blink_time_threshold:  # Verifica se já passou o tempo suficiente
            last_blink_time = time.time()  # Atualiza o tempo da última piscada
            blinks_count += 1  # Aumenta o contador de piscadas
    return last_blink_time, blinks_count

# Enquanto a câmera estiver aberta
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera')
            continue

        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            if not som_tocando:
                pygame.mixer.music.play(-1)
                som_tocando = True
        else:
            if som_tocando:
                pygame.mixer.music.stop()
                som_tocando = False
        
        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame,
                                          face_landmarks,
                                          mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

                face = face_landmarks.landmark
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 255, 0), -1)
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 255, 0), -1)

                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                cv2.rectangle(frame, (0, 1), (290, 140), (58, 58, 55), -1)
                cv2.putText(frame, f"EAR {round(ear, 2)}", (1, 24), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                mar = calculo_mar(face, p_boca)
                cv2.putText(frame, f"MAR: {round(mar, 2)} {'aberto' if mar >= mar_limiar else 'fechado'}", (1, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                
                # Controlar o som baseado na abertura da boca
                som_tocando = controlar_som(mar, som_tocando)

                if ear < ear_limiar:
                    t_inicial = time.time() if dormindo == 0 else t_inicial
                    dormindo = 1  # dormindo
                if dormindo == 1 and ear >= ear_limiar:
                    dormindo = 0  # acordado
                t_final = time.time()

                tempo = (t_final - t_inicial) if dormindo == 1 else 0.0
                cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                # Identificar sonolência
                if tempo >= 1.5:
                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435), cv2.FONT_HERSHEY_DUPLEX, 0.85, (58, 58, 55), 1)

                # Detectar piscadas e atualizar o contador
                last_blink_time, blinks_count = detectar_piscada(ear, last_blink_time, blinks_count)

                # Exibir a contagem de piscadas
                cv2.putText(frame, f"Piscadas: {blinks_count}", (1, 110), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        except Exception as e:
            print("Erro:", e)
        finally:
            print("Processamento concluído")
        
        cv2.imshow('Camera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Fecha a captura
cap.release()
cv2.destroyAllWindows()



# pip install opencv-python
# pip install mediapipe
# pip install pygame
# pip freeze >> requirements.txt