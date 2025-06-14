# main.py

import cv2
from ultralytics import YOLO
from utils.read_save import read_video, save_video
from utils.tracker import ReIDTracker # <-- IMPORT our new Re-ID Tracker

# --- Configuration ---
MODEL_PATH = 'best.pt'
INPUT_VIDEO_PATH = '15sec_input_720p.mp4'
OUTPUT_VIDEO_PATH = 'player_reid_output.mp4'

def main():
    # --- Load Model ---
    model = YOLO(MODEL_PATH)
    
    # --- Initialize Our New Re-ID Tracker ---
    tracker = ReIDTracker(max_disappeared=20, max_distance=75)

    # --- Read Video ---
    cap, fps, frame_size = read_video(INPUT_VIDEO_PATH)
    if cap is None: return

    # --- Setup Video Writer ---
    out = save_video(OUTPUT_VIDEO_PATH, fps, frame_size)

    # --- Process Frames ---
    while True:
        success, frame = cap.read()
        if not success: break

        # Run detection
        results = model(frame, conf=0.5)
        
        # --- Prepare Detections for the Tracker ---
        # The new tracker needs the bounding box for histogram calculation
        detections_for_tracker = []
        for result in results:
            player_classes = ['player', 'goalkeeper'] 
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                if class_name in player_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox = (x1, y1, x2, y2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    centroid = (cx, cy)
                    detections_for_tracker.append({'bbox': bbox, 'centroid': centroid})

        # --- Update Tracker ---
        # We pass the full frame so the tracker can crop player images
        tracked_players = tracker.update(detections_for_tracker, frame)

        # --- Visualize Results ---
        for player_id, player_data in tracked_players.items():
            centroid = player_data['centroid']
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {player_id}", (centroid[0] - 10, centroid[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Re-ID tracking complete! Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()