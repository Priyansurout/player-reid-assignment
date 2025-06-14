# utils/reid_tracker.py

import cv2
import numpy as np
from scipy.spatial.distance import cdist

class ReIDTracker:
    def __init__(self, max_disappeared=20, max_distance=50):
        """
        Initializes our Re-ID tracker.

        Args:
            max_disappeared (int): Max consecutive frames an object can be lost
                                   before being deregistered.
            max_distance (int): Max distance (pixels) to consider a match based
                                on position.
        """
        # --- The Tracker's Memory ---
        self.next_player_id = 0
        self.tracked_players = {}  # {player_id: {'centroid':(x,y), 'hist':hist, ...}}
        
        # --- Configuration ---
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _calculate_histogram(self, frame, bbox):
        """
        Calculates the color histogram for a player's bounding box.
        This serves as the player's "color fingerprint".
        """
        # 1. Crop the player's image from the frame using the bounding box
        x1, y1, x2, y2 = bbox
        player_img = frame[int(y1):int(y2), int(x1):int(x2)]

        # 2. Convert the cropped image to the HSV color space
        hsv_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        
        # 3. Calculate the histogram for the Hue channel
        # We use the Hue channel as it's most representative of pure color.
        # We use 16 bins for the histogram for simplicity.
        hist = cv2.calcHist([hsv_img], [0], None, [16], [0, 180])
        
        # 4. Normalize the histogram to a range of 0-255
        # This makes comparisons more reliable.
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        
        return hist.flatten() # Return the histogram as a flat array

    def register(self, centroid, histogram):
        """Registers a new player with a new ID, centroid, and histogram."""
        player_id = self.next_player_id
        self.tracked_players[player_id] = {
            'centroid': centroid,
            'hist': histogram,
            'disappeared': 0
        }
        self.next_player_id += 1

    def deregister(self, player_id):
        """Deregisters a player who has disappeared."""
        del self.tracked_players[player_id]

    def update(self, detections, frame):
        """
        The main engine of the tracker. Updates the state with new detections.

        Args:
            detections (list): A list of dictionaries, where each dictionary
                               contains the 'bbox' and 'centroid' for a detected player.
            frame (np.array): The full video frame, used for calculating histograms.
        """
        # --- Step 1: Handle cases with no detections or no tracked players ---
        if len(detections) == 0:
            for player_id in list(self.tracked_players.keys()):
                self.tracked_players[player_id]['disappeared'] += 1
                if self.tracked_players[player_id]['disappeared'] > self.max_disappeared:
                    self.deregister(player_id)
            return self.tracked_players

        if len(self.tracked_players) == 0:
            for detection in detections:
                hist = self._calculate_histogram(frame, detection['bbox'])
                self.register(detection['centroid'], hist)
            return self.tracked_players
            
        # --- Step 2: Prepare data for matching ---
        tracked_ids = list(self.tracked_players.keys())
        old_centroids = np.array([p['centroid'] for p in self.tracked_players.values()])
        old_histograms = [p['hist'] for p in self.tracked_players.values()]
        
        new_centroids = np.array([d['centroid'] for d in detections])
        new_histograms = [self._calculate_histogram(frame, d['bbox']) for d in detections]

        # --- Step 3: Perform matching using both distance and appearance ---
        
        # Calculate the distance between all old and new centroids
        dist_matrix = cdist(old_centroids, new_centroids)
        
        # Calculate the histogram similarity between all old and new players
        hist_sim_matrix = np.zeros((len(old_histograms), len(new_histograms)))
        for i in range(len(old_histograms)):
            for j in range(len(new_histograms)):
                # cv2.compareHist with CORREL method gives a value between -1 and 1
                # (1 is a perfect match).
                hist_sim_matrix[i, j] = cv2.compareHist(old_histograms[i], new_histograms[j], cv2.HISTCMP_CORREL)

        # --- A Simple Matching Strategy ---
        # We find the best match for each old player based on distance first,
        # then check if the histogram similarity is good enough.
        
        matched_indices = dist_matrix.argmin(axis=1)
        used_new_indices = set()
        
        for i, player_id in enumerate(tracked_ids):
            best_match_idx = matched_indices[i]
            
            # Check 1: Is the distance acceptable?
            if dist_matrix[i, best_match_idx] < self.max_distance:
                # Check 2: Is the appearance similar enough?
                # We set a threshold, e.g., 0.5 for correlation.
                if hist_sim_matrix[i, best_match_idx] > 0.5:
                    if best_match_idx not in used_new_indices:
                        # Successful match! Update the player.
                        self.tracked_players[player_id]['centroid'] = detections[best_match_idx]['centroid']
                        self.tracked_players[player_id]['disappeared'] = 0
                        used_new_indices.add(best_match_idx)
                        continue # Move to the next tracked player

            # If we reach here, the match failed either by distance or appearance.
            self.tracked_players[player_id]['disappeared'] += 1
            if self.tracked_players[player_id]['disappeared'] > self.max_disappeared:
                self.deregister(player_id)
        
        # --- Step 4: Handle new players ---
        unmatched_new_indices = set(range(len(detections))) - used_new_indices
        for idx in unmatched_new_indices:
            # For this simple Re-ID, we will just register them as new.
            # A more advanced version would check against lost players here.
            self.register(detections[idx]['centroid'], new_histograms[idx])
            
        return self.tracked_players