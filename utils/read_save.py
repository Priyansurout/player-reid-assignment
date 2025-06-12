# utils/video_utils.py

import cv2

def read_video(video_path):
    """
    Reads a video file and returns the capture object and its properties.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing the video capture object, frames per second,
               and frame dimensions (width, height).
        Returns (None, None, None) if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None, None, None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    
    return cap, fps, frame_size

def save_video(output_path, fps, frame_size):
    """
    Creates a VideoWriter object to save a video file.

    Args:
        output_path (str): The path to save the output video file.
        fps (int): The frames per second for the output video.
        frame_size (tuple): The dimensions (width, height) for the output video.

    Returns:
        cv2.VideoWriter: The VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return out