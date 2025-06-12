# main.py

import cv2
from ultralytics import YOLO
from utils.video_utils import read_video, save_video
from utils.tracker import PlayerTracker # <-- IMPORT our new Tracker class

# --- Configuration ---
MODEL_PATH = 'best.pt'
INPUT_VIDEO_PATH = '15sec_input_720p.mp4'
OUTPUT_VIDEO_PATH = 'player_tracking_output.mp4'

def main():
    # --- Load Model ---
    model = YOLO(MODEL_PATH)
    
    # --- Initialize Tracker ---
    tracker = PlayerTracker()

    # --- Read Video ---
    cap, fps, frame_size = read_video(INPUT_VIDEO_PATH)
    if cap is None: return

    # --- Setup Video Writer ---
    out = save_video(OUTPUT_VIDEO_PATH, fps, frame_size)

    # --- Process Frames ---
    while True:
        success, frame = cap.read()
        if not success: break

        # 1. DETECT: Run detection
        results = model(frame)
        
        # 2. EXTRACT CENTROIDS: Get centroids of detected players
        detected_centroids = []
        for result in results:
            # We only care about 'player' and 'goalkeeper'
            player_classes = ['player', 'goalkeeper'] 
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                if class_name in player_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detected_centroids.append((cx, cy))

        # 3. UPDATE TRACKER: Give the new centroids to the tracker
        tracked_players = tracker.update(detected_centroids)

        # 4. VISUALIZE: Draw the tracking results on the frame
        for player_id, centroid in tracked_players.items():
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1) # Draw a red dot at the center
            cv2.putText(frame, f"ID: {player_id}", (centroid[0] - 10, centroid[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Draw the ID

        # Write the frame
        out.write(frame)

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking complete! Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()