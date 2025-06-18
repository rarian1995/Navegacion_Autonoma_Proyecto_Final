from controller import Robot, Camera
from vehicle import Driver
import numpy as np
import tensorflow as tf
import cv2

# Inicializar robot
robot = Robot()
driver = Driver()
timestep = int(robot.getBasicTimeStep())

# Inicializar cámara
camera = robot.getDevice("camera")
camera.enable(timestep)

# Cargar modelo entrenado
model = tf.keras.models.load_model("steering_model.h5")

# Función para preprocesar imagen
def preprocess(image, size=(64, 64)):
    img = np.frombuffer(image, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalizar
    return img

# Configurar velocidad constante
driver.setCruisingSpeed(15)

# Loop principal
while robot.step(timestep) != -1:
    raw_image = camera.getImage()
    if raw_image:
        input_img = preprocess(raw_image)
        input_img = np.expand_dims(input_img, axis=0) 

        # Predecir ángulo
        predicted_angle = float(model.predict(input_img, verbose=0)[0])

        # Aplicar al auto
        driver.setSteeringAngle(predicted_angle)