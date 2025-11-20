from pymavlink import mavutil
from ultralytics import YOLO
import cv2, time, math, json, os
from datetime import datetime

# -------------------- CONFIG --------------------
MAVLINK_CONN = "udp:127.0.0.1:14551"
YOLO_MODEL = "best(S6).pt"

SERVO_CHANNEL = 9
PWM_ON = 1800
PWM_OFF = 1100
TRIGGER_DELAY = 1.5

SMOOTH_DESCENT_ALT = 5     # meters above ground before drop
HOVER_TIME = 2               # seconds to hover before dropping

# -------------------- CONNECT --------------------
print("🔗 Connecting to drone...")
master = mavutil.mavlink_connection(MAVLINK_CONN)
master.wait_heartbeat()
print("✅ Connected to drone.")

model = YOLO(YOLO_MODEL)

cap = cv2.VideoCapture(0)

# ---------------- MAVLINK HELPERS ----------------
def mode_set_and_wait(mode):
    master.set_mode(master.mode_mapping()[mode])
    print(f"⏳ Setting mode to {mode}...")
    for _ in range(50):
        master.recv_match(type="HEARTBEAT", blocking=False)
        if master.flightmode == mode:
            print(f"✅ Mode changed to {mode}")
            return True
        time.sleep(0.2)
    return False

def get_current_location():
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    if msg:
        return (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000)
    return None

def goto_location(lat, lon, alt):
    master.mav.set_position_target_global_int_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b110111111000,
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

def get_distance_meters(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6371000
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2)/2))
    y = math.radians(lat2 - lat1)
    return math.sqrt(x*x + y*y) * R

def trigger_servo(ch, pwm_open, pwm_close, delay):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
        ch, pwm_open, 0, 0, 0, 0, 0
    )
    time.sleep(delay)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
        ch, pwm_close, 0, 0, 0, 0, 0
    )

# ---------------- CAMERA TO GPS MAPPING ----------------
def compute_target_location(cx, cy, w, h):
    # simplified mapping to target point in front of drone
    lat, lon, alt = get_current_location()

    OFFSET = 0.0000089  # approx movement
    tgt_lat = lat + OFFSET * (cy - h/2)/h
    tgt_lon = lon + OFFSET * (cx - w/2)/w

    return tgt_lat, tgt_lon, alt


# ----------------- PAYLOAD DROP SEQUENCE ------------------
def payload_drop(box, frame):

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = (x1 + x2)//2
    cy = (y1 + y2)//2
    h, w = frame.shape[:2]

    tgt_lat, tgt_lon, tgt_alt = compute_target_location(cx, cy, w, h)
    print(f"🎯 Target Coordinates: {tgt_lat}, {tgt_lon}")

    # Switch to GUIDED
    if not mode_set_and_wait("GUIDED"):
        print("❌ Could not switch to GUIDED")
        return False

    print("✈ Navigating to target...")
    goto_location(tgt_lat, tgt_lon, tgt_alt)

    # Arrive check
    while True:
        cur = get_current_location()
        if cur:
            dist = get_distance_meters((cur[0], cur[1]), (tgt_lat, tgt_lon))
            print(f"Distance: {dist:.2f}m   ", end="\r")
            if dist < 1:
                break
        time.sleep(0.3)

    print("\n✅ Arrived above target")

    # ------------------ SMOOTH DESCENT ------------------
    print(f"⬇ Smooth descent to {SMOOTH_DESCENT_ALT}m...")
    goto_location(tgt_lat, tgt_lon, SMOOTH_DESCENT_ALT)

    while True:
        cur = get_current_location()
        if cur:
            if abs(cur[2] - SMOOTH_DESCENT_ALT) < 0.3:
                break
        time.sleep(0.3)

    print("🟢 Reached drop altitude")

    # ---------------- HOVER BEFORE DROP ----------------
    print(f"🛑 Hovering {HOVER_TIME}s before drop...")
    hover_until = time.time() + HOVER_TIME
    while time.time() < hover_until:
        goto_location(tgt_lat, tgt_lon, SMOOTH_DESCENT_ALT)
        time.sleep(0.2)

    # ---------------- VERIFY & DROP ----------------
    print("🔍 Verifying BOX for final confirmation...")
    verify_start = time.time()

    while time.time() - verify_start < 4:
        ret, vframe = cap.read()
        if not ret:
            continue

        results = model(vframe, conf=0.5)
        if results and results[0].boxes:
            for b in results[0].boxes:
                label = model.names[int(b.cls[0])]
                if label.upper() == "BOX":
                    print("📦 Verified! Dropping payload...")
                    trigger_servo(SERVO_CHANNEL, PWM_ON, PWM_OFF, TRIGGER_DELAY)
                    print("✅ Payload dropped successfully!")

                    # ------------- RTL AFTER DROP -------------
                    print("🏠 Switching to RTL...")
                    mode_set_and_wait("RTL")
                    return True

    print("❌ BOX not confirmed. Aborting drop.")
    return False


# ---------------------- MAIN LOOP -------------------------
print("🔍 Starting detection loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=0.5)

    if results and results[0].boxes:
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            if label.upper() == "BOX":
                print("\n📦 BOX detected! Initiating drop sequence...")
                payload_drop(box, frame)
                break

    cv2.imshow("LIVE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
