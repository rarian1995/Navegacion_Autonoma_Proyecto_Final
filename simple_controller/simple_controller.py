from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
import time

# Variables globales
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 15


# Configuración de directorios y archivo CSV para guardar imágenes y datos
session_date = datetime.now().strftime("%Y-%m-%d")
image_dir = os.path.join(os.getcwd(), session_date)
csv_file_path = os.path.join(image_dir, "log.csv")

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "steering_angle"])

# Control del tiempo para guardar imágenes con intervalo definido (1 imagen por segundo)
last_saved_time = 0
save_interval = 1.0  # segundos

# Función para ajustar la velocidad
def set_speed(kmh):
    global speed
    speed = kmh

# Función para suavizar y limitar el ángulo de dirección
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    max_delta = 0.01
    delta = wheel_angle - steering_angle
    if delta > max_delta:
        wheel_angle = steering_angle + max_delta
    elif delta < -max_delta:
        wheel_angle = steering_angle - max_delta
    steering_angle = np.clip(wheel_angle, -0.5, 0.5)
    angle = steering_angle

# Función para cambiar el ángulo de dirección manualmente con límite más pequeño para evitar giros bruscos
def change_steer_angle(inc):
    global manual_steering
    inc = inc * 0.2  # reducir incremento para control más suave
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)

# Función para obtener la imagen del sensor cámara
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Función para mostrar la imagen en el display de Webots solo si el display existe
def display_image(display, image):
    if display is None:
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Función principal que ejecuta el ciclo del robot
def main():
    global last_saved_time

    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = robot.getDevice("display")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        current_time = time.time()

        # Captura y muestra imagen de la cámara
        image = get_image(camera)
        display_image(display_img, image)

        # Guardar imagen y log cada 1 segundo
        if current_time - last_saved_time >= save_interval:
            last_saved_time = current_time

            timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
            filename = f"{timestamp}.png"
            filepath = os.path.join(image_dir, filename)
            camera.saveImage(filepath, quality=100)

            csv_writer.writerow([filename, round(angle, 3)])
            csv_file.flush()

            print(f"Guardada imagen {filename} con ángulo {round(angle,3)}")
        key = keyboard.getKey()
        steering_input = 0

        global manual_steering

        while key != -1:
            if key == keyboard.UP:
                set_speed(speed + 5.0)
                print(f"Velocidad: {speed}")
            elif key == keyboard.DOWN:
                set_speed(speed - 5.0)
                print(f"Velocidad: {speed}")
            elif key == keyboard.RIGHT:
                steering_input = +1
            elif key == keyboard.LEFT:
                steering_input = -1
            key = keyboard.getKey()

        # Cambiar ángulo de dirección según input
        if steering_input != 0:
            change_steer_angle(steering_input)
        else:
            # Auto-centrado suave si no se presiona izquierda/derecha
            if manual_steering > 0:
                manual_steering = max(manual_steering - 0.5, 0)
            elif manual_steering < 0:
                manual_steering = min(manual_steering + 0.5, 0)
            set_steering_angle(manual_steering * 0.02)

        # Aplicar control al vehículo
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

    csv_file.close()

if __name__ == "__main__":
    main()
