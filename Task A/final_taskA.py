# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:40:59 2025

@author: USER
"""

# Assignment Task A
import numpy as np
import cv2

# Load pre-trained Haar cascade model
face_cascade = cv2.CascadeClassifier("face_detector.xml")

# Function to detect brightness
def detect_brightness(frame, threshold  =100):
    # Turn into grayscale first
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the average brightness
    brightness = np.mean(image_gray)
    
    # Compare the average brightness into below two conditions
    if brightness > threshold:
        return "Daytime"
    else:
        return "Nighttime"

# Function to increase the brightness for nighttime video
def increase_brightness_for_nighttime(frame, alpha = 1.0, beta = 50):
    # addWeighted method is used here
    increase_brightness_video = cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
    return increase_brightness_video

# Function to blur all detected faces in the video
def blur_faces_in_video(video_path):
    # Open the input video file
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Failed to open video {video_path}")
        return
    
    # Get video width and height (pixel)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #video fps, default to 30fps if reading failed
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    # generate output filename
    output_filename = "blurred_" + video_path.rsplit(".", 1)[0].split("/")[-1] + ".avi"

     
    # ensures output video have the same resolution and fps
    out = cv2.VideoWriter(output_filename,
                          cv2.VideoWriter_fourcc(*'MJPG'), # this is motion jpeg codec
                          fps,
                          (width, height)) # Frame size must match the input video
    
    # check whether the output video file is created succesfully
    if not out.isOpened():
        print("Failed to open")
   
    # This while loop is to process the video frame-by-frame
    while True:
        # read one frame from the video
        ret, frame = vid.read()
        if not ret:
            break
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        # To loop through all the detected faces
        for (x,y,w,h) in faces:
           # cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0),2)
            # extract the region where the face is located
            face_region = frame[y:y+h, x:x+w]
            # apply gaussian blur to the face region
            blurred = cv2.GaussianBlur(face_region, (35,35), 30)
            # replace the original face region with the blurred one
            frame[y:y+h, x:x+w] = blurred
        #Write the modified frame to the output video
        out.write(frame)
  
    # Release the video after complete the process
    vid.release()
    out.release()
    print(f"Finished blurring faces in {video_path}, saved as {output_filename}")
    return output_filename

# Define a function to overlay the talking.mp4 on the main video
def overlay_video(main_video_path, talking_video_path="talking.mp4"):
    # Open the main video and the talking video
    main_vid = cv2.VideoCapture(main_video_path)
    talking_vid = cv2.VideoCapture(talking_video_path)
    
    # Check if both videos weere successfully opened
    if not main_vid.isOpened() or not talking_vid.isOpened():
        print(f"Failed to open one of the videos: {main_video_path}, {talking_video_path}")
        return
    
    # Get video width and height (pixel)
    width = int(main_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize pip to smaller size
    pip_width = width // 4
    pip_height = height // 4
    
    # Video fps, default to 30fps if reading failed
    fps = main_vid.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    # Generate output filename
    output_filename = "overlayed_" + main_video_path.split(".")[0] + ".avi"
    
    # Set up the video writer to save the output video
    out = cv2.VideoWriter(output_filename,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          fps,
                          (width,height))
    
    # Loop through the frames of two of the video
    while True:
        # Reach frame by frame from each video
        ret_main, main_frame = main_vid.read()
        ret_talk, talk_frame = talking_vid.read()
        
        if not ret_main:
            break # Main video ends
          
        # If talking video ends, loop it to restart again
        if not ret_talk:
            talking_vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # Go to first frame
            ret_talk, talk_frame = talking_vid.read() # Try to read again
            if not ret_talk:
                break
            
       # Resize the talking video frame to fit smaller PIP window 
        small_talk = cv2.resize(talk_frame, (pip_width,pip_height))
        
        '''main_frame[0:pip_height, 0:pip_width] = small_talk
        x = width - pip_width - padding
        y = height - pip_height - padding'''
        
        # Set position of PIP window using padding from the edges
        padding_x = 40  # Horizontal padding 
        padding_y = 100 # Vertical padding
        # Overlay the resized talking video on the main video at specific position
        main_frame[padding_y:pip_height+padding_y, padding_x:pip_width+padding_x] = small_talk
        # Draw a white bounding box around the PIP window
        cv2.rectangle(main_frame, (padding_x,padding_y), (padding_x+pip_width, padding_y+pip_height), (255,255,255),2)
        
        # Write the processed frame into the output video
        out.write(main_frame)
        
    # Release video resources
    main_vid.release()
    talking_vid.release()
    out.release()
    # Print this line when the overlay process is complete
    print(f"Saved overlay video as {output_filename}")
    return output_filename
    
# Function to overlay watermark at the center of the video     
def watermark_center(frame, watermark):
    # Get the height and width of watermark and frame
    wm_h, wm_w, _ = watermark.shape
    frame_h, frame_w, _ = frame.shape

    # Calculate the center coordinates
    x = (frame_w - wm_w) // 2
    y = (frame_h - wm_h) // 2
    
    # Create mask where non-black pixels in the watermark are visible
    # Convert watermark to grayscale
    gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    mask = gray > 10  # Threshold that ignore black background
 
    # Blend watermark onto frame using weigthed averaging
    for c in range(3):  # For each color channel (B, G, R)
        frame[y:y+wm_h, x:x+wm_w, c][mask] = (
            (1 - 0.7) * frame[y:y+wm_h, x:x+wm_w, c][mask] + # Frame pixel * (1 - 0.7 (opacity))
            0.7 * watermark[:, :, c][mask]                   # Watermark pixel * 0.7 (opacity)   
        ).astype(np.uint8)
       
    return frame

# Function to add watermarks in a video frame    
def add_watermarks(video_path, watermark1_path="watermark1.png", watermark2_path="watermark2.png"):
    # Open the video file
    vid = cv2.VideoCapture(video_path)
    
    # Load the watermark image
    watermark1 = cv2.imread(watermark1_path, cv2.IMREAD_UNCHANGED)
    watermark2 = cv2.imread(watermark2_path, cv2.IMREAD_UNCHANGED)
    
    # Retrive frame properties
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    # Calculate the half of total video frame
    half_point = total_frames // 2

    frame_index = 0
    
    # Set output filename
    output_filename = "watermarked_" + video_path
    out = cv2.VideoWriter(output_filename, 
                          cv2.VideoWriter_fourcc(*'MJPG'), 
                          fps, 
                          (width, height))
    
    # Process each frame    
    while True:
        ret, frame = vid.read()
        if not ret:
            # Exit loop when no more frames
            break
        
        # Use watermark1.png on the first half video and watermark2.png on the second half
        current_watermark = watermark1 if frame_index < half_point else watermark2

        # Add the watermark to the center of the current frame
        frame = watermark_center(frame, current_watermark)
        
        # Write the processed frame into the output video
        out.write(frame)
        frame_index += 1

    vid.release()
    out.release()
    print(f"Center watermark added: {output_filename}")
    return output_filename

# Function to add the end screen video to the end of a main video
def append_endscreen(main_video_path, endscreen_path="endscreen.mp4"):
    # Open both the main video and end screen video file
    main_vid = cv2.VideoCapture(main_video_path)
    end_vid = cv2.VideoCapture(endscreen_path)
    

    # Retrive main video frame properties
    width = int(main_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(main_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = main_vid.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    # Set output filename
    output_filename = "final_" + main_video_path
    out = cv2.VideoWriter(output_filename,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          fps,
                          (width, height))

    # Write frames from the main video
    while True:
        ret, frame = main_vid.read()
        if not ret:
            # Exit loop when no more frames
            break
        # Write the main frame into the output video
        out.write(frame)
    
    # Write end screen frames after the main video
    while True:
        ret_end, end_frame = end_vid.read()
        if not ret_end or end_frame is None:
            # Exit loop when no more frames
            break

        # Resize the end screen video to match the main video
        end_frame = cv2.resize(end_frame, (width, height))
        
        # Write the end screen videoframe into the output video
        out.write(end_frame)

    main_vid.release()
    end_vid.release()
    out.release()
    print(f"End screen appended: {output_filename}")
    
    return output_filename

# Process for each video 
def process_video(video_list):
    final = [] 
    # Loop through each video in the video_list
    for video_path in video_list:
        # Open the input file video
        vid = cv2.VideoCapture(video_path)
        # Read the next frame from the video 
        success, frame = vid.read()

        # Show error message if cannot read the video
        if not success:
            print("Failed to read the video: {video_path}")
            # Since cannot read the video, we will then skip to next video
            continue 
        
        # Check whether it is daytime or nighttime video using detect_brightness function
        result = detect_brightness(frame)
        print(f"{video_path}--------{result}")
        
        # If it is nighttime video, proceed here to increase the brightness
        if result == "Nighttime":
            # Get the width and height of the original video frame
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Get the frame rate of the video
            frame_per_second = vid.get(cv2.CAP_PROP_FPS)
            # If the frame rate cannot be read properly, set it default to 30.0
            if frame_per_second == 0:
                frame_per_second = 30.0
            
            # Rewrite the output (brightned video) to a new video file
            output_filename = f"brightened_{video_path.split('.')[0]}.avi"
            out = cv2.VideoWriter(output_filename,                    
                                  cv2.VideoWriter_fourcc(*'MJPG'),          
                                  30.0,                                     
                                  (width, height)) 
            
            # Reset the video to first frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Loop through all of the frames in the video
            while True:
                success, frame = vid.read()
                # If no frame can be read already, it means have reached the end of the video OR there got error while reading the frame
                if not success:
                    break
                # Increase brightness for those nighttime video
                brightness_frame = increase_brightness_for_nighttime(frame)
                # Rewrite to the output video file
                out.write(brightness_frame)
                
            # Close the output video file properly so it can save correctly 
            out.release()
            print(f"Finished increasing brightness in {video_path}, saved as {output_filename}")
            # Always close and release the input video file to avoid any conflict
            vid.release()
        
        # If it is not nighttime video (means is daytime), then execute the following else block part    
        else:
            # Not need to increase brightness for daytime video
            vid.release()
            # Use original path as the output
            output_filename = video_path
        
        # Blur faces in the video
        blurred = blur_faces_in_video(output_filename)

        # Overlay the talking video into the blurred video
        overlayed = overlay_video(blurred)

        # Add watermarks into the overlayed video
        watermarked = add_watermarks(overlayed)

        # Add the end screen into the watermarked video and add the output_filename to the final list 
        final.append(append_endscreen(watermarked))
        
    return final     
        
if __name__ == "__main__":
    
    # List of video
    videos = ["alley.mp4",
              "singapore.mp4",
              "office.mp4",
              "traffic.mp4"]
    
    # Run the processing and get the list of final outputs
    processed = process_video(videos)
    
    # Print out the final output filenames
    print("\nFinal Processed Video Filenames:")
    for v in processed:
        print(v)