import torch
import cv2 as cv
import numpy as np
import os
import torchvision.transforms as transforms
from skimage import io
from typing import Dict, Tuple
import time


class Scaled_Image:
    def __init__(self, image, scale) -> None:
        self.image = image
        self.scale = scale


def create_image_pyramid(image: np.ndarray, scales: list = [1]) -> list:
    pyramid = []
    for scale in scales:
        width, height, _ = image.shape
        new_image = cv.resize(image, (int(scale * height), int(scale * width)))
        pyramid.append(Scaled_Image(new_image, scale))

    return pyramid


def get_patch(image, y_coords: Tuple[int, int], x_coords: Tuple[int, int]) -> np.ndarray:
    y_min, y_max = y_coords
    x_min, x_max = x_coords
    return image[y_min: y_max, x_min: x_max, :]


def sliding_window(image: np.ndarray, window_size: Tuple[int, int], stride: Tuple[int, int]):
    height, width = image.shape[:2]

    window_width = window_size[0]
    window_height = window_size[1]

    horizontal_stride = stride[0]
    vertical_stride = stride[1]
    for y in range(0, height - window_width + 1, horizontal_stride):
        for x in range(0, width - window_height + 1, vertical_stride):
            x_min, y_min = x, y
            x_max, y_max = x + window_height, y + window_width
            yield ((x_min, y_min, x_max, y_max), image[y:y + window_width, x:x + window_height, :])


def prepare_image(image: np.ndarray, transform=None) -> torch.tensor:
    if transform:
        image = transform(image)
    return image


def detect_faces(face_detector, image_pyramid: list, window_size: Tuple[int, int] = (96, 96), stride_ratio: float = 1) -> Tuple[list, list]:
    horizontal_stride = int(stride_ratio * window_size[0])
    vertical_stride = int(stride_ratio * window_size[1])
    stride = (horizontal_stride, vertical_stride)

    face_coords = []
    scores = []

    for scaled_image in image_pyramid:
        # put the model in evaluation mode
        face_detector.eval()
        with torch.no_grad():
            for ((coords), window) in sliding_window(scaled_image.image, window_size, stride):
                # set the trnasforms object to pass it to
                # the prepare_image fucntion

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((96, 96), antialias=False),
                ])

                tensor = prepare_image(window, transform)
                tensor = tensor.unsqueeze(0)

                detection = face_detector(tensor)
                probability, prediction = detection.max(1)
                prediction = prediction.item()
                probability = probability.item()

                if 0.75 <= probability and prediction == 1:
                    new_coords = tuple(np.array(coords) // scaled_image.scale)
                    face_coords.append(new_coords)
                    scores.append(probability)

    return face_coords, scores


def intersection_over_union(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def non_maximal_suppression(detections: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    # sort the indices in descending order
    sorted_indices = np.flipud(np.argsort(scores))

    # arrange the detections and indices based on the sorted indices.
    sorted_detections = detections[sorted_indices]
    sorted_scores = scores[sorted_indices]
    is_maximal = np.ones(len(detections)).astype(bool)

    for i, detection1 in enumerate(sorted_detections[:-1]):
        if is_maximal[i] == True:
            for detection2 in sorted_detections[i + 1:]:
                if is_maximal[i + 1] == True:
                    if intersection_over_union(detection1, detection2) > iou_threshold:
                        is_maximal[i] = False
                    else:
                        c_x = (detection2[0] + detection2[2]) / 2
                        c_y = (detection2[1] + detection2[3]) / 2

                        if detection1[0] <= c_x <= detection2[2] and \
                           detection1[1] <= c_y <= detection2[3]:

                            is_maximal[i + 1] = False

    return sorted_detections[is_maximal], sorted_scores[is_maximal]


def format_results(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # splits the result in the desired structure:
    # (detections, file_names, scores) split into 3 .npy files

    file_names_all_faces = []
    detections_all_faces = []
    scores_all_faces = []

    for file_name in results:
        # check how many faces were detected
        faces_detected = len(results[file_name][0])
        if faces_detected:
            for _ in range(faces_detected):
                file_names_all_faces.append(file_name)

            detections = results[file_name][0]
            scores = results[file_name][1]

            for detection, score in zip(detections, scores):
                detections_all_faces.append(detection)
                scores_all_faces.append(score)

    return np.array(detections_all_faces), \
        np.array(file_names_all_faces), \
        np.array(scores_all_faces)


def get_face_detections(folder_path: str, **kwargs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    classifier = kwargs["model"]
    window_size = kwargs["window_size"]
    stride = kwargs["stride"]

    result = {}
    image_paths = [folder_path + "/" + x for x in os.listdir(folder_path)]
    image_names = [x for x in os.listdir(folder_path)]
    for (image_name, image_path) in zip(image_names, image_paths):
        # read image
        image = io.imread(image_path)

        # create the pyramid of images
        pyramid = create_image_pyramid(image)

        # faces_detected = list of tuples containing the coordinates of the
        # extreme pixels that define the square containing the detected face
        faces, scores = detect_faces(classifier, pyramid, window_size, stride)
        final_faces, final_scores = non_maximal_suppression(
            np.array(faces), np.array(scores))

        result[image_name] = (final_faces, final_scores)

    return result


if __name__ == "__main__":

    # PATHS

    # the path where the model is found
    # works for any model with the input shape 96x96
    model_path = r"./resnet18_flintstones_detection_weights.pth"

    # the directory where the .npy files are saved
    solution_path = ""

    # the path of the directory containing the images for test
    # WARNING -> works only for folders containing .jpg files
    test_dir_path = ""

    # PARAMETERS

    # classifier
    # if torch.cuda.is_available():
    #     classifier = torch.load(model_path)
    # else:
    #     classifier = torch.load(model_path, map_location=torch.device('cpu'))

    classifier = torch.load(model_path, map_location=torch.device('cpu'))

    # sliding window size
    window_size = (128, 96)

    # stride used when moving the window
    stride = 1

    # time tracker
    start_time = time.time()

    print("%% Running face detection ... %%")
    results = get_face_detections(test_dir_path,
                                  model=classifier,
                                  window_size=window_size,
                                  stride=stride)

    end_time = time.time()
    print(f"%% Time elapsed {end_time - start_time} seconds %%")
    print("%% Processing results ... %%")
    detections_all_faces, file_names_all_faces, scores_all_faces = format_results(
        results)

    # if the directory doesn't exist, create it
    if not os.path.exists(solution_path):
        os.makedirs(solution_path)

    np.save(f"{solution_path}detections_all_faces.npy", detections_all_faces)
    np.save(f"{solution_path}file_names_all_faces.npy", file_names_all_faces)
    np.save(f"{solution_path}scores_all_faces.npy", scores_all_faces)
    print("Done")
