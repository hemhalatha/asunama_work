from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import cv2
from ultralytics import YOLO  
from pymavlink import mavutil

# Connect to the drone (e.g., SITL or telemetry)
print("code running...")
vehicle = connect("udp:127.0.0.1:14552", wait_ready=True)
print("drone connected....")
vehicle.mode = VehicleMode("GUIDED")

# Camera specs (in mm and px — tune these)
sensor_width_mm = 7.6      # mm (horizontal sensor size)
focal_length_mm = 4.6      # mm
altitude_m = 10            # will be updated when flying

def arm_and_takeoff(target_altitude):
    print("Arming motors...")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to become armable...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f" Current altitude: {current_alt:.2f}")
        if current_alt >= target_altitude * 0.95:
            print("Target altitude reached!")
            break
        time.sleep(1)


# Convert pixel offset (meters) to GPS location
def get_target_location(current_location, offset_x_m, offset_y_m,alt1):
    R = 6378137.0  # Earth radius in meters

    dLat = offset_y_m / R
    dLon = offset_x_m / (R * math.cos(math.pi * current_location.lat / 180.0))

    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, alt1)

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat * 1.113195e5)**2 + (dlon * 1.113195e5)**2)

def camera_to_uav(x_cam, y_cam):
    x_uav = -y_cam  
    y_uav = x_cam   
    return x_uav, y_uav

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east

# Load YOLO model
model = YOLO('c:/Users/Hemhalatha V R/Downloads/best_model_in_the_world.pt')


# Initialize camera
cap1 = cv2.VideoCapture('rtsp://192.168.144.25:8554/main.264')
cap=cv2.imread(cap1,0)

detected=False

def detect(n):
    global detected
    
    ret, frame = cap.read()
    if not ret:
        print("cant capture image")
        return
    
    frame1=cv2.flip(frame,-1)
    frame1=cv2.resize(frame1,(640,640))
    image_width_px = frame1.shape[1] #(height,width)
    image_height_px = frame1.shape[0]

    # GSD: meters per pixel
    GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * image_width_px)
    
    results = model(frame1)

    if results and results[0].boxes:
        detected=True
        annotated_frame = results[0].plot() 
        cv2.imwrite(f"output_{n}.jpg", annotated_frame)
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

        x_uav, y_uav = camera_to_uav(offset_x_m ,offset_y_m )

    # Convert to North-East using current heading
        yaw_rad = math.radians(vehicle.heading)
        north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)

        print(f"Offset (meters): X = {offset_x_m:.2f}, Y = {offset_y_m:.2f}")

        # Move drone
        current_location = vehicle.location.global_relative_frame
        target_location = get_target_location(current_location, north_offset, east_offset)
        print("current: x:",current_location.lat," y=",current_location.lon)
        print("target: x:",target_location.lat," y=",target_location.lon)    
        distance = get_distance_meters(current_location, target_location)
        print("distance: ",distance)

        print("Navigating to target ",n)
        #trigger_nav=input("enter any key to navigate : ")

        vehicle.simple_goto(target_location)
        while True:
            current_loc = vehicle.location.global_relative_frame
            distance = get_distance_meters(current_loc, target_location)
            if distance < 0.8:
                time.sleep(5)
                break
            time.sleep(1)

    else:
        print("Object not detected")



arm_and_takeoff(15)
while not detected:
    detect(1)    # change the value for image changing

print("Landing...")
trigger_land=input("enter any key to land: ")
vehicle.mode = VehicleMode("LAND")

cap.release()
vehicle.close()
