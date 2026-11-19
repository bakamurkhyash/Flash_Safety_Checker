import cv2
import os
import json
import numpy as np
import time

#listen ro me, create a dictionary and store the frame path alongside all if uts details 
def extract_frames(video_path:str, output_folder="extracted_frames"):

  # Create output directory if it doesn't exist
  os.makedirs(output_folder, exist_ok=True)

  # Open the video file
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
      print("Error: Could not open video.")
      exit()

  frame_count = 0
  frame_paths = []

  while True:
      # Read a frame from the video
      ret, frame = cap.read()
    
      # If ret is False, the video has ended
      if not ret:
          break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      mean_luma = np.mean(gray). 
      frame_luminance.append(mean_luma)
    
      # Save the frame as an image file
      frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
      cv2.imwrite(frame_filename, frame). 
      frame_paths.append(output_folder+"/"+frame_filename)
    
      frame_count += 1

  # Release resources
  cap.release()
  print(f"Done! Extracted {frame_count} frames to '{output_folder}'.")
  return frame_paths
