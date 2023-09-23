# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:05:26 2023

@author: teoan
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift


def dif(arr):
    d=max(arr)-min(arr)
    return d

def create_bandpass_mask(shape, inner_radius, outer_radius):
    mask = np.zeros(shape, dtype=bool)
    center = (shape[0] // 2, shape[1] // 2)

    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if inner_radius <= distance <= outer_radius:
                mask[i, j] = 1

    return mask

def rotate_array(arr, angle):
    radians = np.deg2rad(angle)
    sin_val = np.sin(radians)
    cos_val = np.cos(radians)

    num_rows, num_cols = arr.shape
    center_row, center_col = num_rows / 2, num_cols / 2

    rotated_arr = np.zeros_like(arr)

    for i in range(num_rows):
        for j in range(num_cols):
            translated_row = i - center_row
            translated_col = j - center_col

            rotated_row = round(translated_row * cos_val - translated_col * sin_val + center_row)
            rotated_col = round(translated_row * sin_val + translated_col * cos_val + center_col)

            if 0 <= rotated_row < num_rows and 0 <= rotated_col < num_cols:
                rotated_arr[i, j] = arr[rotated_row, rotated_col]

    return rotated_arr



# Read the video
video_path = "/Users/name/Desktop/video.avi"  #input a video of verical fringes moving
cap = cv2.VideoCapture(video_path)            #open the video

# Get the total number of frames and frame rate
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the duration of the video in seconds
duration_seconds = total_frames / fps

# Parameters for peak detection
peak_height = 100  # Adjust this threshold to detect peaks. Anything above this specified height will be considered a peak
min_peak_distance = 10  # Adjust this value to control minimum peak separation. Anything clsoer than this will be measured as one peak

# List to store peak positions for each frame
all_peaks = []

current_second = 0
image_extracted = False


roi_width_start = 300       #Specify the area of interest in the video. By making this smaller the process becomes faster.
roi_width_end = 1150

roi_height_start = 100
roi_height_end = 950

o=0

while cap.isOpened():
    ret, frame = cap.read()
    
    if  ret:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        current_second = int(current_time)
        
        '''The below if statement is placed to extarct an image every 2 seconds instead 
        for every frame in the video. If the fringes are moving slow we can extract less images to 
        analyse and save time. If the fringes shift fast then we need to utilize more frames from the video.'''
        if not image_extracted and current_second % 2 == 0:
            
            # Preprocess the frame (convert to grayscale, apply filters, etc.)
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            scaled_frame = (grayscale_frame.astype(np.float32) / 255) * 1023  #turn from 255 values (8bit) to 1022(12bit)
        
            int_frame = scaled_frame.astype(np.int32)
            
            roi = int_frame[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
    
            # plt.imshow(roi, cmap='gray')  #show the roi for visual inspection
            # plt.axis('off')
            # plt.show()
            
            
            height, width = roi.shape
            
            '''The below for loops aim to rotate the images extracted to make the fringes verical if the video ones are not already verical. 
            In the for loop with the half_rot variable we can indicate the angles to search for and it will run through them and automatically
            choose the best one. This process takes a lot of time so the range should be as small as possible and thus we should start with as 
            verical as possible fringes in the video. This process can definetly get faster processing since right now it ittorates for every pixel
            which is not optimal but for the purpose of the process this worked fine so it was kept as is.'''
            dif_array=[]
            for half_rot in range(1,3,1):
                new_array=rotate_array(roi, half_rot)
                colvalue=[]
                for col in range(0,width,1):
                    colint=0
                    for row in range(0,height,1):
                        colint= colint+ new_array[row,col]
                    colvalue.append(colint)
                dif_array.append((dif(colvalue),half_rot))
                
            for g in range(len(dif_array)):
                if dif_array[g]==max(dif_array):
                    break      
            
            final_array=rotate_array(roi, dif_array[g][1])
            height = final_array.shape[0]
            width=final_array.shape[1]
            
            
            '''The below steps inverse fourier transform the image, create a mask that can have both inner and outer radius, apply it on the image
            to elliminate sertain parts (high frequency components ans such, must be adjusted per use) and then fourier transform again to
            get back a 'smoothed' image '''
            image_fft = fftshift(fft2(final_array))
            mask_shape = (height, width)
            inner_radius = 0
            outer_radius = 70
        
            mask = create_bandpass_mask(mask_shape, inner_radius, outer_radius)
        
            filtered_image_fft = image_fft *mask
            filtered_image = np.abs(ifft2(filtered_image_fft))
            
            '''columnvalue is a variable that has all the values in a column summed up. 
            This is needed since if the fringes are verical we can locate their position by finding the peak of this value.'''
            columnvalue=[]
            for columnsum in range(0,width,1):
                columnint=0
                for row in range(0,height,1):
                    columnint= columnint+ filtered_image[row,columnsum]
                columnvalue.append(columnint)
        
            #remove the low value components that serve as noise
            for row2 in range(0,len(columnvalue),1):
                if columnvalue[row2]<27000:            #Sets a lower limit to eliminate lower component noise in the fft
                    columnvalue[row2]=23000
                    
            x = np.linspace(0, len(columnvalue),len(columnvalue))
            y= np.array(columnvalue)
        
            # Perform peak detection on the preprocessed frame
            peaks, _ = find_peaks(y, height=peak_height, distance=min_peak_distance)
             
            '''The below if unhashed create a plot of the columnvalue variable and annotate the detected peaks or
            otherwise called the fringe positions'''
            # plt.plot(x, y)
            # plt.scatter(x[peaks], y[peaks], c='red', marker='x')
            # for peak in peaks:
            #     plt.annotate(f'Peak {peak}', (x[peak], y[peak]), xytext=(3, 3),textcoords='offset points') #textcoords='offset points')
            # plt.legend([f'{o}'])
            # plt.show()
            print(o)
            o=o+1           #this o variable is used to track in what image the program is, since the code takes a long time to run depending on the parameters
            
            '''all_peaks stores all the peak positions for all the images analysed. This is the variable that in the second part of the
            code we use to track the peak position moving'''
            # Store the detected peaks
            all_peaks.append((current_second,peaks))  
            image_extracted = True
            
            
        if current_second % 2 != 0:
            image_extracted = False
            
    else:
        break
        
cap.release()

# Initialize the peak_tracks dictionary to store tracks
peak_tracks = {}


'''This code was designed for videos that the fringes move in one direction then they
 stop moving for some time and then move again on the opposite direction. This was to create the hysteresis 
 plots which entale moving in one direction and then the other. The below variables dictate the timestamps
 to help the code track the fringes correctly. from 0 to the 1st timestamp (time1st) the fringes move to the left,
 from time1st and until time2nd the fringes are expected to not move but just be unstable due to noise and
 from time2nd to time3rd the fringes are expected to move on the opposite direction (right)'''
#Set the end time (in seconds) of the 1st direction movement, the end time of the creep and the end time of the opposite direction movement
time1st=62
time2nd = 182
time3rd = 432

# Specify the minimum peak position difference from frame to frame
min_peak_position_difference = 0

# Specify the maximum allowed distance for matching peaks
max_distance = 20


def is_match3(peak1, peak2):
    time1 = peak1[0]
    peak_pos1 = peak1[1][1]
    time2= peak2[0]
    peak_pos2 = peak2[1]
    
    
    if time2 <= time1st:
        if abs(peak_pos2 - peak_pos1) <= 4:    #this acound for the instability/ shake of the fringes (if the fringes are shaking a lot 4 must become higher or more images being used from the video)
            return abs(peak_pos2 - peak_pos1) <= 4
        if abs(peak_pos2 - peak_pos1)>4:
            
            # Peaks move to lower positions/ left
            return peak_pos1 - peak_pos2 >= min_peak_position_difference
    if time1st<time2 <= time2nd:
        return abs(peak_pos2 - peak_pos1) < max_distance
    if time2nd<time2 <= time3rd:
        if abs(peak_pos2 - peak_pos1) <= 4:
            return abs(peak_pos2 - peak_pos1) <= 4 #this acound for the instability/ shake of the fringes (if the fringes are shaking a lot 4 must become higher or more images being used from the video)
        if abs(peak_pos2 - peak_pos1)>4:
            # Peaks move to higher positions/ right
            return peak_pos2 - peak_pos1 >= min_peak_position_difference
    if time3rd<time2 :
        return abs(peak_pos2 - peak_pos1) < max_distance



# Iterate over each frame and track the peaks
for frame_idx, (time ,peaks) in enumerate(all_peaks):
    if frame_idx == 0:
        # For the first frame, create tracks for all peaks
        for peak_idx, peak_pos in enumerate(peaks):
            peak_tracks[peak_idx] = [(frame_idx,(time, peak_pos))]
    else:
        # Create a list to keep track of matched peaks
        matched_peaks = []

        # Create a set to keep track of used tracks
        used_tracks = set()

        # Iterate over peaks in the current frame
        for peak_idx, peak_pos in enumerate(peaks):
            matching_track_idx = None
            min_distance = 26         #here place a pixel value that is slightly higher than the expected distnace of the fringes.

            # Iterate over existing tracks to find the closest match
            for track_idx, track in peak_tracks.items():
                last_peak = (track[-1][0],track[-1][1])

                # Check if the peak satisfies proximity and minimum position difference criteria
                if (
                    track_idx not in used_tracks
                    and is_match3(last_peak, ((time,peak_pos)))
                
                ):
                    distance = abs(peak_pos - last_peak[1][1])

                    if distance < min_distance:
                        matching_track_idx = track_idx
                        min_distance = distance

            # If a matching track is found, update the track
            if matching_track_idx is not None:
                peak_tracks[matching_track_idx].append((frame_idx, (time,peak_pos)))
                matched_peaks.append(matching_track_idx)
                used_tracks.add(matching_track_idx)
            else:
                # Create a new track for unmatched peaks
                new_track_idx = max(peak_tracks.keys()) + 1
                peak_tracks[new_track_idx] = [(frame_idx, (time,peak_pos))]
                matched_peaks.append(new_track_idx)

        # Remove unmatched tracks
        unmatched_tracks = set(peak_tracks.keys()) - set(matched_peaks)
        for track_idx in unmatched_tracks:
            del peak_tracks[track_idx]


# # Print the peak tracks
# for track_idx, track in peak_tracks.items():
#     print(f"Track {track_idx}:")
#     for frame_idx, peak_pos in track:
#         print(f"  Frame {frame_idx}: Peak Position {peak_pos}")



# Specify the track index you want to print
# track_index = 11

# if track_index in peak_tracks:
#     track = peak_tracks[track_index]
#     print(f"Track {track_index}:")
#     for frame_idx, peak_pos in track:
#         print(f"  Frame {frame_idx}: Peak Position {peak_pos}")
# else:
#     print(f"No track found with index {track_index}")