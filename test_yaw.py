from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import cv2
from ultralytics import YOLO  # Assuming you use YOLOv8
from pymavlink import mavutil

# Connect to the drone
vehicle = connect('127.0.0.1:14550', wait_ready=True)

# Parameters (adjust GSD based on your setup)
GSD = 0.05  # meters per pixel — tune this

# Arm and takeoff
def arm_and_takeoff(aTargetAltitude):
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

# Convert pixel offset to GPS
def get_target_location(current_location, offset_x, offset_y):
    R = 6378137.0
    dLat = offset_y / R
    dLon = offset_x / (R * math.cos(math.pi * current_location.lat / 180))

    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, current_location.alt)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start process
arm_and_takeoff(10)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    if results and results[0].boxes:
        box = results[0].boxes[0]
        xmin, ymin, xmax, ymax = box.xyxy[0]
        object_centerx = int((xmin + xmax) / 2)
        object_centery = int((ymin + ymax) / 2)

        # Calculate image center
        image_centerx = frame.shape[1] // 2
        image_centery = frame.shape[0] // 2

        # Pixel offset
        offset_x = (object_centerx - image_centerx) * GSD
        offset_y = (object_centery - image_centery) * GSD

        print(f"Offset (m): X={offset_x}, Y={offset_y}")

        # Move drone
        current_location = vehicle.location.global_relative_frame
        target_location = get_target_location(current_location, offset_x, offset_y)
        print("Navigating to target...")
        vehicle.simple_goto(target_location)

        time.sleep(10)  # Wait to reach
        print("Landing...")
        vehicle.mode = VehicleMode("LAND")
        break

cap.release()
vehicle.close()