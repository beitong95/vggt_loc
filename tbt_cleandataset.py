import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# usage example:
# 1. creat datasets folder under vggt
# 2. create a experiment folder under datasets folder (e.g. new3113)
# 3. put train/test iphone/rpi data under exp folder
# 4. set train_test to "train" or "test" to process train or test data
# 5. set exp_name to the experiment folder name
# 6. set dataset_iphone_path and dataset_rpi_path to the iphone and rpi data path
# 7. run the script
# 8. check the exp_name_cleaned folder for the cleaned data
# 9. check the output video under exp_name folder to see if they are synced
# do the above for both train and test data

# Config 1 
train_test = "test"
OUTPUT_VIDEO = True
dataset_base_path = "/data6/beitong2/vggt/datasets"
exp_name = "new3113"


dataset_exp_path = os.path.join(dataset_base_path, exp_name)
output_video_path = os.path.join(dataset_exp_path, f"output_side_by_side_{train_test}.mp4")

# Config 2: change iphone and rpi path respectively
# train
if train_test == "train":
    dataset_iphone_path = os.path.join(dataset_exp_path, "testlong_2025-03-23_19-08-25")
    dataset_rpi_path = os.path.join(dataset_exp_path, "testlong_2025-03-23_19-08-25_rpi")
# test
else:
    dataset_iphone_path = os.path.join(dataset_exp_path, "testlong_2025-03-23_19-10-35")
    dataset_rpi_path = os.path.join(dataset_exp_path, "testlong_2025-03-23_19-10-35_rpi")

dataset_iphone_rgb_path = os.path.join(dataset_iphone_path, "rgb")
dataset_iphone_poses_path = os.path.join(dataset_iphone_path, "poses")
dataset_rpi_rgb_path = dataset_rpi_path

# step 1:
# create result folders
# folder struct
# - exp_name_cleaned
#   - train
#     - iphone
#       - rgb
#       - calibration
#       - poses
#     - rpi
#       - rgb
#   - test ... (same as train)

output_exp_path = os.path.join(dataset_base_path, f"{exp_name}_cleaned")
os.makedirs(output_exp_path, exist_ok=True)
output_dataset_path = os.path.join(output_exp_path, train_test)
os.makedirs(output_dataset_path, exist_ok=True)
output_dataset_iphone_path = os.path.join(output_dataset_path, "iphone")
os.makedirs(output_dataset_iphone_path, exist_ok=True)
output_dataset_iphone_rgb_path = os.path.join(output_dataset_iphone_path, "rgb")
os.makedirs(output_dataset_iphone_rgb_path, exist_ok=True)
output_dataset_iphone_calibration_path = os.path.join(output_dataset_iphone_path, "calibration")
os.makedirs(output_dataset_iphone_calibration_path, exist_ok=True)
output_dataset_iphone_poses_path = os.path.join(output_dataset_iphone_path, "poses")
os.makedirs(output_dataset_iphone_poses_path, exist_ok=True)
output_dataset_rpi_path = os.path.join(output_dataset_path, "rpi")
os.makedirs(output_dataset_rpi_path, exist_ok=True)
output_dataset_rpi_rgb_path = os.path.join(output_dataset_rpi_path, "rgb")
os.makedirs(output_dataset_rpi_rgb_path, exist_ok=True)
output_dataset_rpi_poses_path = os.path.join(output_dataset_rpi_path, "poses")
os.makedirs(output_dataset_rpi_poses_path, exist_ok=True)

def compute_redness_score(region):
    region = region.astype(np.float32)
    R = region[:, :, 2]
    G = region[:, :, 1]
    B = region[:, :, 0]
    redness = R - np.maximum(G, B)
    redness = np.clip(redness, 0, 255)
    return np.mean(redness)

def process_image_folder_diff_prev(folder_path, diff_threshold=4):
    redness_scores = []
    detections = []
    filenames = sorted(os.listdir(folder_path))[50:] # skip first 50
    fps_iphone = 30
    flash_interval = 5 # seconds
    min_interval = fps_iphone * flash_interval - 5
    estimated_count = int(len(filenames) / (fps_iphone * flash_interval))
    print(f"Estimated count: {estimated_count}")
    res = []
    res_index = []
    

    prev_score = None
    for fname in filenames:
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        fname_no_ext = os.path.splitext(fname)[0]
        fname_image_index = int(fname_no_ext.split("_")[0])
        fname_time = int(fname_no_ext.split("_")[1])
        fname_status = fname_no_ext.split("_")[2]

        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        cropped = img[40:160, -120:]  # Top-right corner
        score = compute_redness_score(cropped)
        redness_scores.append(score)

        if prev_score is None:
            detections.append(0)
        else:
            diff = score - prev_score
            detections.append(diff)
            if diff > diff_threshold:
                if len(res_index) > 0 and fname_image_index < res_index[-1] + min_interval:
                    continue
                print(f"Red laser detected in {fname}")
                res.append(fname)
                res_index.append(fname_image_index)

        prev_score = score
    print(f"Result sync frame = {len(res_index)}")

    return res

if os.path.exists(os.path.join(dataset_exp_path, f"iphone_sync_image_name_list_{train_test}.npy")):
    iphone_sync_image_name_list = np.load(f"iphone_sync_image_name_list_{train_test}.npy")
else:
    iphone_sync_image_name_list = process_image_folder_diff_prev(dataset_iphone_rgb_path)
    np.save(os.path.join(dataset_exp_path, f"iphone_sync_image_name_list_{train_test}.npy"), iphone_sync_image_name_list)

filenames = sorted(os.listdir(dataset_rpi_rgb_path))
filepaths = [os.path.join(dataset_rpi_rgb_path, f) for f in filenames if f.endswith('.jpg')]


# step 2:
iphone_rgb_names = os.listdir(dataset_iphone_rgb_path)
iphone_rgb_names = [f for f in iphone_rgb_names if f.endswith('.jpg') and "normal" in f] # skip not normal since only normal gives us correct pose
iphone_rgb_names = sorted(iphone_rgb_names)
iphone_tracking_stats = {}
all_iphone_time_stamps = []
for i, name in enumerate(iphone_rgb_names):
    iphone_tracking_status = name.split("_")[2].split(".")[0]
    iphone_time_stamp = int(name.split("_")[1])
    all_iphone_time_stamps.append(iphone_time_stamp)
    if iphone_tracking_status not in iphone_tracking_stats:
        iphone_tracking_stats[iphone_tracking_status] = 1
    else:
        iphone_tracking_stats[iphone_tracking_status] += 1
print(f"iphone_tracking_stats: {iphone_tracking_stats}")

# step 3:
rpi_rgb_names = os.listdir(dataset_rpi_rgb_path)
rpi_rgb_names = [f for f in rpi_rgb_names if f.endswith('.jpg')]
rpi_rgb_names = sorted(rpi_rgb_names)
rpi_rgb_stats = {}
for i, name in enumerate(rpi_rgb_names):
    rpi_rgb_stats[name] = "no match"

reference_rpi_name = ""
prev_time_sync_slot = 0
iphone_search_index = 0
all_map_delta_time = []
for i, name in enumerate(rpi_rgb_names):
    index = name.split("_")[0]
    time_stamp = int(name.split("_")[1])
    time_sync_slot = int(name.split("_")[2].split(".")[0])
    if time_sync_slot == 0:
        continue
    if time_sync_slot != prev_time_sync_slot:
        reference_rpi_name = time_stamp
    prev_time_sync_slot = time_sync_slot

    # find closest iphone name
    delta_rpi_time = int(time_stamp) - int(reference_rpi_name)
    reference_iphone_name = iphone_sync_image_name_list[time_sync_slot-1]
    iphone_time_stamp = reference_iphone_name.split("_")[1]
    target_iphone_time = int(iphone_time_stamp) + delta_rpi_time
    # print(target_iphone_time)
    for j in range(iphone_search_index, len(iphone_rgb_names)):
        if j == len(iphone_rgb_names) - 1:
            rpi_rgb_stats[name] = iphone_rgb_names[j]
        else:
            if abs(all_iphone_time_stamps[j] - target_iphone_time) < abs(all_iphone_time_stamps[j+1] - target_iphone_time):
                rpi_rgb_stats[name] = iphone_rgb_names[j]
                iphone_search_index = j
                all_map_delta_time.append(abs(all_iphone_time_stamps[j] - target_iphone_time))
                break
            else:
                continue

counter = 0
all_count = 0
cleaned_res = {}
for key, value in rpi_rgb_stats.items():
    if int(key.split("_")[2].split(".")[0]) == 0:
        continue
    if value != "no match":
        cleaned_res[key] = value
        counter += 1
    all_count += 1
print(f"Matched: {counter}/{all_count}")
print(f"Stats of delta time: {np.mean(all_map_delta_time)}, Max: {np.max(all_map_delta_time)}, Min: {np.min(all_map_delta_time)}")        


# step 4: copy files:
# rpi
for key, value in cleaned_res.items():
    pose = value.split(".")[0] + ".txt"
    pose_path = os.path.join(dataset_iphone_poses_path, pose)
    if not os.path.exists(pose_path):
        print(f"Warning: Could not find pose file for {value}, skipping.")
    shutil.copy2(pose_path, os.path.join(output_dataset_rpi_poses_path, key.split(".")[0] + ".txt"))
    shutil.copy2(os.path.join(dataset_rpi_rgb_path, key), os.path.join(output_dataset_rpi_rgb_path, key))
# iphone
for name in iphone_rgb_names:
    if "normal" not in name:
        continue
    pose = name.split(".")[0] + ".txt"
    pose_path = os.path.join(dataset_iphone_poses_path, pose)
    if not os.path.exists(pose_path):
        print(f"Warning: Could not find pose file for {name}, skipping.")
    shutil.copy2(pose_path, os.path.join(output_dataset_iphone_poses_path, name.split(".")[0] + ".txt"))
    shutil.copy2(os.path.join(dataset_iphone_rgb_path, name), os.path.join(output_dataset_iphone_rgb_path, name))
    
print(f"iphone {train_test} rgb: {len(os.listdir(output_dataset_iphone_rgb_path))}")
print(f"iphone {train_test} poses: {len(os.listdir(output_dataset_iphone_poses_path))}")
print(f"rpi {train_test} rgb: {len(os.listdir(output_dataset_rpi_rgb_path))}")
print(f"rpi {train_test} poses: {len(os.listdir(output_dataset_rpi_poses_path))}")

# step 5: output videos
if OUTPUT_VIDEO:

    print(f"Outputting side-by-side video to {output_video_path}")

    # Sort dictionary by key
    sorted_items = sorted(cleaned_res.items())

    # Prepare the video writer
    frame_width = 0
    frame_height = 0
    fps = 10  # Set to 1 FPS, adjust as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi

    # Determine frame size from first image pair
    for key, value in sorted_items:
        img1 = cv2.imread(os.path.join(dataset_rpi_rgb_path, key))
        img2 = cv2.imread(os.path.join(dataset_iphone_rgb_path, value))
        if img1 is not None and img2 is not None:
            height = max(img1.shape[0], img2.shape[0])
            width = img1.shape[1] + img2.shape[1]
            frame_height, frame_width = height, width
            break

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Generate video frames
    for key, value in sorted_items:
        path1 = os.path.join(dataset_rpi_rgb_path, key)
        path2 = os.path.join(dataset_iphone_rgb_path, value)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"Warning: Could not load {key} or {value}, skipping.")
            continue

        # Resize to same height if necessary
        height = max(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (img1.shape[1], height))
        img2 = cv2.resize(img2, (img2.shape[1], height))

        # Pad width to match frame width if needed
        img1 = cv2.copyMakeBorder(img1, 0, 0, 0, frame_width//2 - img1.shape[1], cv2.BORDER_CONSTANT)
        img2 = cv2.copyMakeBorder(img2, 0, 0, 0, frame_width//2 - img2.shape[1], cv2.BORDER_CONSTANT)

        combined = cv2.hconcat([img1, img2])
        out.write(combined)

    out.release() 
    print("Done.")
        






