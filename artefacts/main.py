from segmentation_client import SegmentationClient
import pandas as pd
import numpy as np
import math
import cv2 as cv
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv 
import os

def load_and_segment_pdf(pdf_path, column_numbers, main_dir):
    sgc = SegmentationClient(main_dir)
    return [sgc.pdf_scan_to_cells_of_columns(pdf_path, col_nr) for col_nr in column_numbers]

def process_images(table, bird_cnn, number_cnn):
    csvSpecies = classify_column(table[0], bird_cnn, classify_species)
    csvNr = classify_column(table[1], number_cnn, classify_numbers)
    return csvSpecies, csvNr

def classify_column(column_data, cnn_model, classify_func):
    results = []
    for pNr, page in enumerate(column_data):
        for rNr, row in enumerate(page):
            result = classify_func(row, cnn_model)
            results.append([pNr, rNr, result])
    return results

def classify_species(row_image, bird_cnn):
    species_img = prepare_image_for_cnn(row_image, target_size=(22, 150))
    species_prediction = bird_cnn.predict(species_img)
    return get_species_name(species_prediction)


def prepare_image_for_cnn(row_image, target_size):
    data = cv.bitwise_not(row_image)
    cv.imwrite("tmp.png", row_image)
    img = image.load_img("tmp.png", target_size=target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array / 255.0, axis=0)

def get_species_name(species_prediction, species_json_path='class_indices.json'):
    species_index = np.argmax(species_prediction, axis=1)[0]
    with open(species_json_path, 'r') as file:
        class_indices = json.load(file)
    species_dict = {v: k for k, v in class_indices.items()}
    return species_dict.get(species_index, "Unknown")

def classify_numbers(row_image, number_cnn):
    data = preprocess_number_image(row_image)
    return detect_and_classify_digits(data, number_cnn)

def preprocess_number_image(row_image):
    data = cv.bitwise_not(row_image)
    data = remove_noise_and_increase_contrast(data)
    return data

def remove_noise_and_increase_contrast(data):
    components, labeledImage, componentStats, _ = cv.connectedComponentsWithStats(data, connectivity=4)
    minArea = 40
    remaining_labels = [i for i in range(1, components) if componentStats[i][4] >= minArea]
    return np.where(np.isin(labeledImage, remaining_labels), 255, 0).astype('uint8')

def detect_and_classify_digits(data, number_cnn):
    contours, _ = cv.findContours(data, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    sorted_bounding_boxes = [cv.boundingRect(c) for c in contours if cv.contourArea(c) > 10]
    sorted_bounding_boxes.sort(key=lambda x: x[0])
    return classify_bounding_boxes(sorted_bounding_boxes, data, number_cnn)

def classify_bounding_boxes(bounding_boxes, data, number_cnn):
    digits = ""
    for x, y, w, h in bounding_boxes:
        if is_valid_digit_bounding_box(w, h):
            cropped_img = make_square_image(data[y:y + h, x:x + w], size=28)
            digits += str(predict_digit(cropped_img, number_cnn))
    return digits

def is_valid_digit_bounding_box(w, h):
    return (w > h * 0.5) and (w > 10 and h > 10)

def make_square_image(cropped_img, size):
    max_dim = max(cropped_img.shape)
    square_img = np.zeros((max_dim, max_dim))
    offset = (max_dim - cropped_img.shape[0]) // 2
    square_img[offset:offset + cropped_img.shape[0], :cropped_img.shape[1]] = cropped_img
    return cv.resize(square_img, (size, size))

def predict_digit(cropped_img, number_cnn):
    cropped_img = np.expand_dims(cropped_img, axis=(0, -1)) / 255.0
    prediction = number_cnn.predict(cropped_img)
    return np.argmax(prediction)

def save_results(csvSpecies, csvNr, species_output, numbers_output):
    pd.DataFrame(csvSpecies, columns=["Seite", "Zeile", "Vogelart"]).to_csv(species_output, index=False)
    pd.DataFrame(csvNr, columns=["Seite", "Zeile", "Zahl"]).to_csv(numbers_output, index=False)

def main():
    main_dir = os.getenv('MAIN_DIR', '/Users/MeinNotebook/Google Drive/Meine Ablage/Scans')
    test_file = main_dir + "/1972/scan_1972_CdB_2_20231125160645.pdf"
    #MAIN_DIRECTORY = "/content/drive/MyDrive/Scans"
    #TEST_FILE = MAIN_DIRECTORY + "/1972/scan_1972_CdB_2_20231125160645.pdf"
    
    
    #species_model_path = '/content/drive/MyDrive/Scans/Models/vogelarten_best_model.keras'
    #number_model_path = '/content/drive/MyDrive/Scans/Models/mnist_cnn_model.keras'
    #class_indices_path = '/content/drive/MyDrive/Scans/class_indices.json'
    #species_output = '/content/drive/MyDrive/Scans/predictions.csv'
    #numbers_output = '/content/drive/MyDrive/Scans/predictions_numbers.csv'
    
    species_model_path = os.getenv('SPECIES_KERAS_DIR', 'vogelarten_best_model.keras')
    number_model_path = os.getenv('MNIST_KERAS_DIR', 'mnist_cnn_model.keras')

    pdf_path = test_file

    bird_cnn = load_model(species_model_path)
    number_cnn = load_model(number_model_path)
    species_output = os.getenv('SPECIES_OUTPUT_DIR', '/Users/MeinNotebook/Desktop/predictions.csv')
    numbers_output = os.getenv('NUMBERS_OUTPUT_DIR', '/Users/MeinNotebook/Desktop/predicitons_numbers.csv')
    
    
    ### for local testskript only 
    pdf_path = "scan_1972_CdB_10_20231125162253.pdf"
    ###
    table = load_and_segment_pdf(pdf_path, [1, 6], main_dir)
    csvSpecies, csvNr = process_images(table, bird_cnn, number_cnn)
    save_results(csvSpecies, csvNr, species_output, numbers_output)

if __name__ == "__main__":
    load_dotenv()
    main()
