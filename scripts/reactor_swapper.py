import copy
import os
import shutil
from dataclasses import dataclass
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

import insightface

from scripts.reactor_logger import logger
from reactor_utils import move_path

import warnings
import time


np.warnings = warnings
np.warnings.filterwarnings('ignore')

providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
print("models_path_old", models_path_old)
insightface_path_old = os.path.join(models_path_old, "insightface")
print("insightface_path_old", insightface_path_old)
insightface_models_path_old = os.path.join(insightface_path_old, "models")
print("insightface_models_path_old", insightface_models_path_old)

models_path = "models"
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")



if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
    
# if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
#     shutil.rmtree(insightface_path_old)
#     shutil.rmtree(models_path_old)

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODEL = None


def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path, download=False,
        )
    return ANALYSIS_MODEL


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    face_gender = gender[face_index]
    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.info("OK - Detected Gender matches Condition")
        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.info("WRONG - Detected Gender doesn't match Condition")
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 1


# def reget_face_single(img_data, det_size, face_index):
#     det_size_half = (det_size[0] // 2, det_size[1] // 2)
#     return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

def half_det_size(det_size):
    logger.info("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser.get(img_data)

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0):

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            # return reget_face_single(img_data, det_size, face_index)
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target)
        return get_face_gender(face,face_index,gender_source,"Source")

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target)
        return get_face_gender(face,face_index,gender_target,"Target")
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target)

    try:
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
         return face[face_index], 0
    except IndexError:
        return None, 0

def compute_face_data(faces):
    return [
        {
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "pos": (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)
        }
        for bbox in (face['bbox'] for face in faces)
    ]

def match_face_index(source_faces, target_faces, source_img, target_img):
    face_data = compute_face_data(source_faces)
    target_data = compute_face_data(target_faces)

    max_faces = 4

    if len(face_data) == 0:
        # if there's no faces in the reference image
        print("No Input Faces found to swap")
        return None

    elif len(face_data) == 1:
        # if there's one face in the reference image
        print("Single reference Face found, returning index 0 for 4 largest target faces")
        largest_target_indices = sorted(range(len(target_data)), key=lambda x: target_data[x]["area"], reverse=True)[:max_faces]
        indexes = [0 if i in largest_target_indices else None for i in range(len(target_data))]
        return indexes

    else:
        # if there are multiple faces in the reference image, match them to the nearest face in the target image, limited to the 4 biggest faces
        x_scaling_factor = target_img.shape[1] / source_img.shape[1]  # width
        y_scaling_factor = target_img.shape[0] / source_img.shape[0]  # height

        scaled_face_data = [{
            "area": face["area"] * x_scaling_factor * y_scaling_factor,  # Area is scaled by both factors
            "pos": (face["pos"][0] * x_scaling_factor, face["pos"][1] * y_scaling_factor)
        } for face in face_data]

        # Get the indices of the 4 largest target faces
        largest_target_indices = sorted(range(len(target_data)), key=lambda x: target_data[x]["area"], reverse=True)[:max_faces]
        indexes = [None] * len(target_data)

        for idx in largest_target_indices:
            target_face = target_data[idx]
            distances = [
                (x["pos"][0] - target_face["pos"][0])**2 + (x["pos"][1] - target_face["pos"][1])**2
                for x in scaled_face_data
            ]
            closest_index = distances.index(min(distances))
            indexes[idx] = closest_index

            # Removed the code that makes the matched face unmatchable

        return indexes


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
):
    result_image = target_img

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        logger.info("Analyzing Source Image...")
        source_faces = analyze_faces(source_img)

        if source_faces is not None:

            logger.info("Analyzing Target Image...")
            target_faces = analyze_faces(target_img)
            
            matched_indexes = match_face_index(source_faces, target_faces, source_img, target_img)
            
            print("Matched Indexes:", matched_indexes)

            faces_index = matched_indexes

            result = target_img.copy()  # Create a copy to avoid altering the original image
            model_path = os.path.join(insightface_path, model)
            face_swapper = getFaceSwapModel(model_path)
            
            if faces_index is not None:
                for target_idx, source_idx in enumerate(faces_index):
                    if source_idx is None:
                        continue

                    # Fetch source face using the matched source index
                    source_face, wrong_gender = get_face_single(source_img, source_faces, face_index=source_idx, gender_source=gender_source)

                    if source_face is not None and wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=target_idx, gender_target=gender_target)
                        
                        if target_face is not None and wrong_gender == 0:
                            result = face_swapper.get(result, target_face, source_face)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                            return result_image
                        else:
                            logger.info(f"No target face found for {target_idx}")
                    elif wrong_gender == 1:
                        wrong_gender = 0
                        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        return result_image
                    else:
                        logger.info(f"No source face found for face number {source_idx}.")

                # Debugging section for drawing bounding boxes
                if False:
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                    for target_idx, source_idx in enumerate(matched_indexes):
                        if source_idx is not None:
                            color = colors[source_idx % len(colors)]  # Use color based on source face index
                            
                            # Draw target face bounding box with thickness of 2
                            target_bbox = target_faces[target_idx]['bbox']
                            print(f"Drawing bounding box for target face {target_idx}: bbox={target_bbox}, color={color}")  # Debugging line
                            cv2.rectangle(result, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), color, 2)

                            # Label the target face bounding box with its index
                            label_position = (int(target_bbox[0])+20, int(target_bbox[1]) + 20)  # Positioning the label just above the bounding box
                            cv2.putText(result, str(target_idx), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                            # Draw scaled source face bounding box on the target image with a thickness of 5
                            source_bbox = source_faces[source_idx]['bbox']
                            scaled_bbox = [
                                int(source_bbox[0] * target_img.shape[1] / source_img.shape[1]),
                                int(source_bbox[1] * target_img.shape[0] / source_img.shape[0]),
                                int(source_bbox[2] * target_img.shape[1] / source_img.shape[1]),
                                int(source_bbox[3] * target_img.shape[0] / source_img.shape[0])
                            ]
                            print(f"Drawing scaled bounding box for source face {source_idx}: bbox={scaled_bbox}, color={color}")  # Debugging line
                            cv2.rectangle(result, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), color, 5)



            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        else:
            logger.info("No source face(s) found")

    return result_image
