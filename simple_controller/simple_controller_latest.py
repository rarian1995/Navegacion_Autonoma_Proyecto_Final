from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

# Variables globales para dirección y velocidad
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20

# Establece la velocidad del vehículo
def set_speed(kmh):
    global speed
    speed = kmh

# Actualiza el ángulo de dirección con límites
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    elif (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = np.clip(wheel_angle, -0.5, 0.5)
    angle = steering_angle

# Captura una imagen de la cámara y la convierte en un arreglo de NumPy
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Define una región de interés trapezoidal en la mitad inferior de la imagen
def get_roi_vertices(image_shape):
    height, width = image_shape[:2]
    bottom_offset = height
    top_height = int(0.75 * height)
    left_top = int(0.05 * width)
    right_top = int(0.95 * width)

    vertices = np.array([[ 
        (0, bottom_offset),
        (left_top, top_height),
        (right_top, top_height),
        (width, bottom_offset)
    ]], dtype=np.int32)

    return vertices

# Procesamiento de imagen: escala de grises, desenfoque, Canny, ROI y detección de líneas
def greyscale_cv2(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Isolate white and yellow lanes
    white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply mask and grayscale
    masked_image = cv2.bitwise_and(image, image, mask=lane_mask)
    gray_img = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)
    vertices = get_roi_vertices(img_canny.shape)
    img_roi = np.zeros_like(gray_img)
    cv2.fillPoly(img_roi, vertices, 255)
    img_mask = cv2.bitwise_and(img_canny, img_roi)

    # Detección de líneas con la Transformada de Hough
    rho = 1
    theta = np.pi/180
    treshold = 30
    min_line_len = image.shape[1] * 0.05
    max_line_gap = 20
    lines = cv2.HoughLinesP(img_mask, rho, theta, treshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Dibuja las líneas detectadas en color blanco
    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), [255, 255, 255], 2)

    # Superpone las líneas detectadas sobre la imagen original
    img_lane_lines = cv2.addWeighted(img_rgb, 0.8, img_lines, 1, 0)
    return img_lane_lines, lines

# Función para detectar carriles y calcular el ángulo de dirección
def detect_lanes(image, lines):
    height, width, _ = image.shape
    left_xs, right_xs = [], []

    if lines is None:
        return 0.0  # Si no hay líneas detectadas, ir recto

    # Clasifica las líneas según su pendiente
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # evita división por cero
        slope = (y2 - y1) / (x2 - x1)
        if 0.3 < abs(slope) < 3:  # ignora líneas horizontales
            if slope < 0:
                left_xs.extend([x1, x2])
            else:
                right_xs.extend([x1, x2])

    # Calcula el centro del carril
    center_x = width // 2
    left_x = np.mean(left_xs) if left_xs else center_x - 60
    right_x = np.mean(right_xs) if right_xs else center_x + 60
    lane_center = (left_x + right_x) / 2

    # Calcula el ángulo de dirección (normalizado entre -1 y 1)
    steering_angle = (lane_center - center_x) / center_x
    return steering_angle

# Función para mostrar la imagen procesada en el display de Webots
def display_image(display, image):
    image_rgb = image.copy()
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Función principal que controla el robot
def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        image = get_image(camera)
        processed_img, lines = greyscale_cv2(image)
        display_image(display_img, processed_img)  # Now correctly calls the display_image function

        # Calcular ángulo de dirección y aplicarlo
        steering_angle = detect_lanes(image, lines)
        set_steering_angle(steering_angle * 0.5)

        # Control desde el teclado
        key = keyboard.getKey()
        if key == keyboard.UP:
            set_speed(speed + 5.0)
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
        elif key == keyboard.LEFT:
            change_steer_angle(-1)
        elif key == ord('A'):
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
            camera.saveImage(os.path.join(os.getcwd(), filename), 1)

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()