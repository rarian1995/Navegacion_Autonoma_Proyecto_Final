import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from controller import Robot, Camera, Lidar
from vehicle import Driver
import numpy as np
import tensorflow as tf
import cv2

robot = Robot()
driver = Driver()
timestep = int(robot.getBasicTimeStep())

# Dispositivos
camera = robot.getDevice("camera")
camera.enable(timestep)

lidar = robot.getDevice("front_lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

# Modelo de CNN para dirección
model = tf.keras.models.load_model("steering_model.h5")

# Parámetros
max_speed = 15
safe_distance_vehicle = 10.0
safe_distance_pedestrian = 5.0
current_speed = max_speed
last_angle = 0.0

def preprocess(image, size=(64, 64)):
    img = np.frombuffer(image, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

driver.setCruisingSpeed(current_speed)

while robot.step(timestep) != -1:
    # Cámara
    raw_image = camera.getImage()
    if raw_image:
        input_img = preprocess(raw_image)
        input_img = np.expand_dims(input_img, axis=0)
        predicted_angle = model.predict(input_img, verbose=0)[0][0].item()
        predicted_angle = np.clip(predicted_angle, -0.5, 0.5)
        smoothed_angle = 0.8 * last_angle + 0.2 * predicted_angle
        driver.setSteeringAngle(smoothed_angle)
        last_angle = smoothed_angle

    # Lidar
    lidar_values = lidar.getRangeImage()
    center_index = len(lidar_values) // 2
    distance_center = lidar_values[center_index]

    # Verificar distancia para detenerse (vehículos o peatones)
    if distance_center < safe_distance_pedestrian:
        current_speed = 0
        print(f"[!] Objeto detectado a {distance_center:.2f} m. Deteniendo.")
    elif distance_center < safe_distance_vehicle:
        current_speed = 0
        print(f"[!] Vehículo detectado a {distance_center:.2f} m. Deteniendo.")
    else:
        current_speed = max_speed

    driver.setCruisingSpeed(current_speed)
