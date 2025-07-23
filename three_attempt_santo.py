from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import cv2
from ultralytics import YOLO  
from pymavlink import mavutil

# Connect to the drone (e.g., SITL or telemetry)
print("sdfgh")
vehicle = connect("com3", wait_ready=True)
print("started")

# Camera specs (in mm and px — tune these)
sensor_width_mm = 7.6      # mm (horizontal sensor size)
focal_length_mm = 4.4      # mm
altitude_m = 10            # will be updated when flying

# Convert pixel offset (meters) to GPS location
def get_target_location(current_location, offset_x_m, offset_y_m):
    R = 6378137.0  # Earth radius in meters
    dLat = offset_y_m / R
    dLon = offset_x_m / (R * math.cos(math.pi * current_location.lat / 180.0))

    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, current_location.alt)

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat * 1.113195e5)**2 + (dlon * 1.113195e5)**2)

# Load YOLO model
model = YOLO('c:/Users/Hemhalatha V R/Downloads/best_model_in_the_world.pt')


# Initialize camera
cap = cv2.VideoCapture('rtsp://192.168.144.25:8554/main.264')

def detect():
    ret, frame = cap.read()
    if not ret:
        print("cant capture image")
        return

    image_width_px = frame.shape[1] #(height,width)
    image_height_px = frame.shape[0]

    # GSD: meters per pixel
    GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * image_width_px)
    
    results = model(frame)

    if results and results[0].boxes:
        detected=True
        annotated_frame = results[0].plot()
        cv2.imwrite("detected_output5.jpg", annotated_frame)
        box = results[0].boxes[0]
        xmin, ymin, xmax, ymax = box.xyxy[0]

        object_centerx = int((xmin + xmax) / 2)
        object_centery = int((ymin + ymax) / 2)

        # Center of the image
        image_centerx = image_width_px // 2
        image_centery = image_height_px // 2

        # Pixel offset → real-world offset (in meters)
        offset_x_m = (object_centerx - image_centerx) * GSD
        offset_y_m = (object_centery - image_centery) * GSD

        print(f"Offset (meters): X = {offset_x_m:.2f}, Y = {offset_y_m:.2f}")

        # Move drone
        current_location = vehicle.location.global_relative_frame
        target_location = get_target_location(current_location, offset_x_m, offset_y_m)
        print("current: x:",current_location.lat," y=",current_location.lon)
        print("target: x:",target_location.lat," y=",target_location.lon)    
        distance = get_distance_meters(current_location, target_location)
        print("distance: ",distance)

    else:
        print("Object not detected")
        





detect()


cap.release()
vehicle.close()
