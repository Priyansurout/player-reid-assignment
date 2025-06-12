# utils/tracker.py

import numpy as np
from scipy.spatial.distance import cdist

class PlayerTracker:
    def __init__(self, max_disappeared=20, max_distance=50):
        """
        Initializes the tracker.

        Args:
            max_disappeared (int): The maximum number of consecutive frames a
                                   player can be temporarily lost before being
                                   deregistered.
            max_distance (int): The maximum distance (in pixels) between centroids
                                to consider them a match.
        """
        # --- The Tracker's Memory ---
        self.next_player_id = 0
        self.tracked_players = {}  # {player_id: centroid}
        self.disappeared_frames = {}  # {player_id: num_of_frames}

        # --- Configuration ---
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Registers a new player with a new ID."""
        player_id = self.next_player_id
        self.tracked_players[player_id] = centroid
        self.disappeared_frames[player_id] = 0
        self.next_player_id += 1

    def deregister(self, player_id):
        """Deregisters a player who has disappeared."""
        del self.tracked_players[player_id]
        del self.disappeared_frames[player_id]

    def update(self, detected_centroids):
        """
        Updates the tracker with the new set of detected centroids for a frame.

        Args:
            detected_centroids (list): A list of (x, y) centroids detected
                                       in the current frame.

        Returns:
            dict: The dictionary of currently tracked players.
        """
        # If there are no detections, mark all tracked players as disappeared
        if len(detected_centroids) == 0:
            for player_id in list(self.disappeared_frames.keys()):
                self.disappeared_frames[player_id] += 1
                if self.disappeared_frames[player_id] > self.max_disappeared:
                    self.deregister(player_id)
            return self.tracked_players

        # If we are not tracking anyone yet, register all new detections
        if len(self.tracked_players) == 0:
            for centroid in detected_centroids:
                self.register(centroid)
            return self.tracked_players

        # --- The Matching Logic ---
        
        # 1. Prepare the data for matching
        tracked_ids = list(self.tracked_players.keys())
        old_centroids = np.array(list(self.tracked_players.values()))
        new_centroids = np.array(detected_centroids)

        # 2. Calculate the distance matrix
        # Rows = old players, Columns = new detections
        distances = cdist(old_centroids, new_centroids)

        # 3. Find the best match for each old player
        # Find the smallest value in each row
        matched_indices = distances.argmin(axis=1)
        matched_distances = distances.min(axis=1)

        # --- Update, Register, and Deregister based on matches ---
        used_new_indices = set()
        
        # Loop through old players and their best matches
        for i, player_id in enumerate(tracked_ids):
            match_idx = matched_indices[i]
            
            # If the match is within our distance threshold and is not already used
            if matched_distances[i] < self.max_distance and match_idx not in used_new_indices:
                # This is a successful match, update the player's position
                self.tracked_players[player_id] = detected_centroids[match_idx]
                self.disappeared_frames[player_id] = 0 # Reset disappeared counter
                used_new_indices.add(match_idx)
            else:
                # This player was not matched, mark it as disappeared
                self.disappeared_frames[player_id] += 1
                if self.disappeared_frames[player_id] > self.max_disappeared:
                    self.deregister(player_id)
        
        # Find which new detections were not used
        all_new_indices = set(range(len(detected_centroids)))
        unmatched_new_indices = all_new_indices - used_new_indices

        # Register all unmatched new detections as new players
        for idx in unmatched_new_indices:
            self.register(detected_centroids[idx])

        return self.tracked_players