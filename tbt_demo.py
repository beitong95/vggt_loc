import gradio as gr
import os
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

# tbt
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb, load_glb_and_update_camera, load_glb_and_update_cameras
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


RETRIEVE_COUNT = 2
# Import CLIP and DINOv2
import clip
from transformers import AutoModel, AutoImageProcessor

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# global
glbfile = ""
scene_scale = 1.0
predictions = {}
predictions_test = {}
current_index = 0
current_test_index = 0
# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

descriptor_cache = {}
image_folder = None
descriptor_method = "DINOv2"
previous_camera_position = None
previous_gt_camera_position = None



device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)



def compute_descriptor(image: Image.Image, method="DINOv2"):
    image_tensor = T.Resize((224, 224))(T.ToTensor()(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        if method == "CLIP":
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            descriptor = clip_model.encode_image(image_input)
        elif method == "DINOv2":
            inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
            descriptor = dinov2_model(**inputs).last_hidden_state.mean(dim=1)
        elif method == "DINOv2+CLIP":
            # Combine DINOv2 and CLIP descriptors
            inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
            dino_desc = dinov2_model(**inputs).last_hidden_state.mean(dim=1)

            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            clip_desc = clip_model.encode_image(image_input)

            descriptor = torch.cat([dino_desc, clip_desc], dim=1)
        else:
            raise ValueError("Invalid method")

    return descriptor.squeeze(0).cpu()

def load_folder(exp_name, folder, method, progress=gr.Progress(track_tqdm=True)):
    global descriptor_cache, image_folder, descriptor_method
    image_folder = folder
    descriptor_method = method
    pickle_path = os.path.join(folder, f"{exp_name}_descriptors_{method}.pkl")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            descriptor_cache = pickle.load(f)
        return f"Loaded descriptors from {pickle_path}"
    
    descriptor_cache = {}
    image_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for filename in progress.tqdm(image_files, desc="Computing descriptors"):
        path = os.path.join(folder, filename)
        image = Image.open(path).convert("RGB")
        desc = compute_descriptor(image, method)
        descriptor_cache[filename] = desc

    with open(pickle_path, "wb") as f:
        pickle.dump(descriptor_cache, f)

    return f"Computed and saved descriptors to {pickle_path}"


def retrieve_similar(query_image):
    if not descriptor_cache:
        return [None]*5, ["No descriptors found. Load image folder first."]*5

    query_desc = compute_descriptor(query_image, descriptor_method)

    similarities = []
    for filename, desc in descriptor_cache.items():
        sim = torch.nn.functional.cosine_similarity(
            query_desc.unsqueeze(0), desc.unsqueeze(0)
        ).item()
        similarities.append((filename, sim))

    top5 = sorted(similarities, key=lambda x: -x[1])[:2]
    images = [Image.open(os.path.join(image_folder, fn)) for fn, _ in top5]
    paths = [os.path.join(image_folder, fn) for fn, _ in top5]
    captions = [f"{fn} (sim={sim:.4f})" for fn, sim in top5]

    return images, captions, paths

def camera_center_from_extrinsic(E):
    R = E[:3, :3]
    t = E[:3, 3]
    center_world = -R.T @ t
    return center_world

def to_homogeneous(E_3x4):
    return np.vstack([E_3x4, np.array([[0, 0, 0, 1]])])

def get_opencv_conversion_matrix() -> np.ndarray:
    # Create an identity matrix
    matrix = np.identity(4)
    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix

def opengl_to_opencv_extrinsic_w2c(E_gl):
    M = get_opencv_conversion_matrix()
    E_flipped = M @ E_gl
    return E_flipped

def opengl_to_opencv_extrinsic_c2w(E_gl):
    M = get_opencv_conversion_matrix()
    E_flipped = E_gl @ M
    return E_flipped

def compute_scale(t1, t2, t1_gt, t2_gt):
    return np.linalg.norm(t2_gt - t1_gt) / np.linalg.norm(t2 - t1)

def update_camera():
    global glbfile, scene_scale, predictions, current_index, previous_camera_position, previous_gt_camera_position
    
    glbscene, previous_camera_position, previous_gt_camera_position = load_glb_and_update_camera(glbfile, predictions, current_index, scene_scale, previous_camera_position, previous_gt_camera_position)   
    current_index += 1
    current_index = min(current_index, len(predictions["extrinsic"]) - 1)
    glbscene.export(file_obj=glbfile)
    return glbfile

from PIL import Image, ImageDraw, ImageFont
import os

def save_retrieved_images(img1, img2, img3, path, translation_error, rotation_error_deg, runtime_ms):
    # Resize all images to the same height
    min_height = min(img1.height, img2.height, img3.height)
    img1 = img1.resize((int(img1.width * min_height / img1.height), min_height))
    img2 = img2.resize((int(img2.width * min_height / img2.height), min_height))
    img3 = img3.resize((int(img3.width * min_height / img3.height), min_height))

    # Concatenate images side-by-side
    total_width = img1.width + img2.width + img3.width
    text_height = 50  # Extra space on top for text
    concat_img = Image.new('RGB', (total_width, min_height + text_height), (255, 255, 255))  # White background

    # Paste images below the text line
    concat_img.paste(img1, (0, text_height))
    concat_img.paste(img2, (img1.width, text_height))
    concat_img.paste(img3, (img1.width + img2.width, text_height))

    # Prepare text
    t_err_rounded = round(translation_error, 2)
    r_err_rounded = round(rotation_error_deg, 2)
    text = f"Translation Error: {t_err_rounded} m | Rotation Error: {r_err_rounded}° | Runtime: {runtime_ms} ms"

    # Draw text
    draw = ImageDraw.Draw(concat_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Calculate text position using textbbox (better than textsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (total_width - text_width) // 2
    draw.text((text_x, 10), text, fill="black", font=font)
    base, ext = os.path.splitext(path)
    new_filename = f"{base}_T{t_err_rounded}_R{r_err_rounded}{ext}_runtime{runtime_ms}.jpg"

    # Save image
    concat_img.save(new_filename)


def compute_camera_error(extrinsic1, extrinsic2):
    # Extract rotation and translation
    R1 = extrinsic1[:3, :3]
    t1 = extrinsic1[:3, 3]
    
    R2 = extrinsic2[:3, :3]
    t2 = extrinsic2[:3, 3]
    
    # Translation error (L2 norm)
    translation_error = np.linalg.norm(t1 - t2)
    
    # Rotation error
    # Compute relative rotation matrix
    R_rel = R1 @ R2.T
    # Convert to rotation vector (angle-axis), then get angle in degrees
    rotvec = R.from_matrix(R_rel).as_rotvec()
    rotation_error_deg = np.linalg.norm(rotvec) * (180.0 / np.pi)
    
    return translation_error, rotation_error_deg


def run_test(folder_test, step, skip_step_radio):
    global glbfile, scene_scale, predictions, predictions_test, current_test_index, previous_camera_position, previous_gt_camera_position
    predictions_test = {}
    data_dir = folder_test
    poses_dir = folder_test.replace("rgb", "poses")
    poses_dir_iphone = poses_dir.replace("rpi", "iphone").replace("test", "train")
    image_names = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])[current_test_index:current_test_index+step]
    poses_names = sorted([os.path.join(poses_dir, f.replace("jpg", "txt")) for f in os.listdir(data_dir) if f.endswith('.jpg')])[current_test_index:current_test_index+step]
    res_pred = []
    res_gt = []
    retrieved_images = []
    retrieved_captions = []
    glbfile_test = ""
    source_image = Image.open(image_names[0])
    for image_name, poses_name in zip(image_names, poses_names):
        query_image_pil = Image.open(image_name)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        images, captions, retrieved_paths = retrieve_similar(query_image_pil)
        # retrieved_images.extend(images)
        # retrieved_captions.extend(captions)
        retrieved_images = (images)
        image_name_only = os.path.basename(image_name).split(".")[0]
        retrieved_captions = (captions)
        print(retrieved_paths)
        gt_poses_paths = [os.path.join(poses_dir_iphone, os.path.basename(p).replace("jpg", "txt").replace("rgb", "poses")) for p in retrieved_paths]
        gt_poses_paths.append(poses_name)
        retrieved_paths.append(image_name)
        print(gt_poses_paths)
        all_images = retrieved_paths
        print(all_images)

        images = load_and_preprocess_images(all_images).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                
                if skip_step_radio == "Run":
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                end_time.record()
                torch.cuda.synchronize() # RuntimeError: CUDA error: device not ready
                
                predictions_test = {}
                predictions_test["images"] = images
                predictions_test["extrinsic"] = extrinsic
                predictions_test["intrinsic"] = intrinsic
                if skip_step_radio == "Run":
                    predictions_test["depth_map"] = depth_map
                for key in predictions_test.keys():
                    if isinstance(predictions_test[key], torch.Tensor):
                        predictions_test[key] = predictions_test[key].cpu().numpy().squeeze(0)  # remove batch dimension

                E_list = predictions_test["extrinsic"]
                gt_poses = []
                for gt_pose_name in gt_poses_paths:
                    E_gt = np.loadtxt(gt_pose_name)
                    E_gt_cv = opengl_to_opencv_extrinsic_c2w(E_gt)
                    E_gt = np.linalg.inv(E_gt_cv)
                    gt_poses.append(E_gt)
                predictions_test["gt_poses"] = gt_poses
                res_gt.append(gt_poses[-1])

            
                s = compute_scale(camera_center_from_extrinsic(E_list[0]), camera_center_from_extrinsic(E_list[1]), camera_center_from_extrinsic(predictions_test["gt_poses"][0]), camera_center_from_extrinsic(predictions_test["gt_poses"][1]))
                for E in E_list:
                    E[:3, 3] *= s

                
                W = np.linalg.inv(to_homogeneous(E_list[0])) @ predictions_test["gt_poses"][0]
                
                E_pred_list = []
                for E in E_list:
                    E_pred_list.append(to_homogeneous(E) @ W)
                
                res_pred.append(E_pred_list[-1][:3, :])

                for i in range(len(predictions_test["extrinsic"])):
                    predictions_test["extrinsic"][i] = E_pred_list[i][:3, :]

                if skip_step_radio == "Run":
                    predictions_test["depth_map"] = s * predictions_test["depth_map"]
            
                    world_points = unproject_depth_map_to_point_map(predictions_test["depth_map"], predictions_test["extrinsic"], predictions_test["intrinsic"])
                    predictions_test["world_points_from_depth"] = world_points
                    frame_filter = "All"
                    conf_thres = 50
                    mask_black_bg = False
                    mask_white_bg = False
                    show_cam = True 
                    mask_sky = False
                    prediction_mode = "Depthmap and Camera Branch"   
                    target_dir = "/data6/beitong2/vggt/output"
                    glbfile_test = os.path.join(
                        target_dir,
                        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_for_test.glb",
                    )
                    glbscene, _ = predictions_to_glb(
                        predictions_test,
                        conf_thres=conf_thres,
                        filter_by_frames=frame_filter,
                        mask_black_bg=mask_black_bg,
                        mask_white_bg=mask_white_bg,
                        show_cam=show_cam,
                        mask_sky=mask_sky,
                        target_dir=target_dir,
                        prediction_mode=prediction_mode,
                    )
                    glbscene.export(file_obj=glbfile_test)
                    
                    tmp_dir = "/data6/beitong2/vggt/tmp"
                    glbfile_test_tmp = os.path.join(
                        tmp_dir,
                        f"{image_name_only}_glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_for_test.glb",
                    )
                    glbscene.export(file_obj=glbfile_test_tmp)
            error_pred_e = res_pred[-1]
            error_gt_e = res_gt[-1][:3, :]
            runtime_ms = int(start_time.elapsed_time(end_time))
            translation_error, rotation_error_deg = compute_camera_error(error_pred_e, error_gt_e)
            save_retrieved_path = os.path.join("/data6/beitong2/vggt/tmp", f"{image_name_only}_{"_".join(captions)}.jpg")
            save_retrieved_images(query_image_pil, retrieved_images[0], retrieved_images[1], save_retrieved_path, translation_error, rotation_error_deg, runtime_ms)

    predictions_test = {}
    predictions_test["extrinsic"] = np.array(res_pred)
    print(predictions_test["extrinsic"].shape)
    predictions_test["gt_poses"] = np.array(res_gt)
    print(predictions_test["gt_poses"].shape)
    scene_3d, previous_camera_position, previous_gt_camera_position = load_glb_and_update_cameras(glbfile, predictions, predictions_test, scene_scale, previous_camera_position, previous_gt_camera_position)
    scene_3d.export(file_obj=glbfile)
    current_test_index += step
    print(retrieved_images)
    print(retrieved_captions)
    if skip_step_radio == "Run":
        return glbfile, glbfile_test, source_image, *retrieved_images, "query image", *retrieved_captions
    else:
        return glbfile, glbfile, source_image, *retrieved_images, "query image", *retrieved_captions
        
def run_reconstruct(folder_reconstruct):
    global glbfile, scene_scale, predictions
    predictions = {}
    data_dir = folder_reconstruct
    poses_dir = folder_reconstruct.replace("rgb_reconstruct", "poses")
    image_names = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])
    poses_names = sorted([os.path.join(poses_dir, f.replace("jpg", "txt")) for f in os.listdir(data_dir) if f.endswith('.jpg')])
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            predictions = {}
            predictions["images"] = images
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            predictions["depth_map"] = depth_map
            
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

            E_list = predictions["extrinsic"]
            gt_poses = []
            for gt_pose_name in poses_names:
                E_gt = np.loadtxt(gt_pose_name)
                E_gt_cv = opengl_to_opencv_extrinsic_c2w(E_gt)
                E_gt = np.linalg.inv(E_gt_cv)
                gt_poses.append(E_gt)
            predictions["gt_poses"] = gt_poses
            
            s = compute_scale(camera_center_from_extrinsic(E_list[0]), camera_center_from_extrinsic(E_list[1]), camera_center_from_extrinsic(predictions["gt_poses"][0]), camera_center_from_extrinsic(predictions["gt_poses"][1]))
            for E in E_list:
                E[:3, 3] *= s
            
            W = np.linalg.inv(to_homogeneous(E_list[0])) @ predictions["gt_poses"][0]
            
            E_pred_list = []
            for E in E_list:
                E_pred_list.append(to_homogeneous(E) @ W)
            
            for i in range(len(predictions["extrinsic"])):
                predictions["extrinsic"][i] = E_pred_list[i][:3, :]
            
            predictions["depth_map"] = s * predictions["depth_map"]
            
            world_points = unproject_depth_map_to_point_map(predictions["depth_map"], predictions["extrinsic"], predictions["intrinsic"])
            predictions["world_points_from_depth"] = world_points
            frame_filter = "All"
            conf_thres = 50
            mask_black_bg = False
            mask_white_bg = False
            show_cam = False 
            mask_sky = False
            prediction_mode = "Depthmap and Camera Branch"   
            target_dir = "/data6/beitong2/vggt/output"
            glbfile = os.path.join(
                target_dir,
                f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
            )
            glbscene, scene_scale = predictions_to_glb(
                predictions,
                conf_thres=conf_thres,
                filter_by_frames=frame_filter,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cam,
                mask_sky=mask_sky,
                target_dir=target_dir,
                prediction_mode=prediction_mode,
            )
            glbscene.export(file_obj=glbfile)
            return glbfile


            
with gr.Blocks() as demo:
    with gr.Row():
        exp_name = gr.Textbox(label="Experiment Name", value="exp1")
        folder_input = gr.Textbox(label="Train Database", value="/data6/beitong2/vggt/datasets/new3113_cleaned/train/iphone/rgb")
        method_input = gr.Radio(choices=["DINOv2", "CLIP", "DINOv2+CLIP"], value="DINOv2", label="Descriptor Method")
        load_button = gr.Button("Load & Compute Descriptors")

    load_output = gr.Textbox(label="Status")

    with gr.Row():
        query_image = gr.Image(type="pil", label="Query Image")
        search_button = gr.Button("Retrieve Top-5 Similar Images")
    
    with gr.Row():
        with gr.Column(scale=2):
            folder_test = gr.Textbox(label="Reconstruct Folder", value="/data6/beitong2/vggt/datasets/new3113_cleaned/test/rpi/rgb")
            step_input = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Step")
            skip_step_radio = gr.Radio(choices=["Skip", "Run"], value="Skip", label="Run Step")
            run_button = gr.Button("Run All")
        with gr.Column(scale=4): 
            test_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

    with gr.Row():
        test_reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

    with gr.Row():
        with gr.Column(scale=2):
            folder_reconstruct = gr.Textbox(label="Reconstruct Folder", value="/data6/beitong2/vggt/datasets/new3113_cleaned/train/iphone/rgb_reconstruct_small")
            reconstruct_button = gr.Button("Reconstruct")
            update_button = gr.Button("Update Camera")
        with gr.Column(scale=4):
            reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)
    

    
    with gr.Row():
        sim_images = [gr.Image(label=f"Image {i+1}") for i in range(RETRIEVE_COUNT+1)]
    with gr.Row():
        sim_labels = [gr.Label() for _ in range(RETRIEVE_COUNT+1)]

    load_button.click(fn=load_folder, inputs=[exp_name, folder_input, method_input], outputs=load_output)
    search_button.click(fn=retrieve_similar, inputs=query_image, outputs=sim_images + sim_labels)
    reconstruct_button.click(fn=run_reconstruct, inputs=folder_reconstruct, outputs=reconstruction_output)
    update_button.click(fn=update_camera, outputs=reconstruction_output)
    run_button.click(fn=run_test, inputs=[folder_test, step_input, skip_step_radio], outputs=[test_output, test_reconstruction_output] + sim_images + sim_labels)
demo.launch()
