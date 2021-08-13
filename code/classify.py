from dataset import DemoDataset
from dataset import encode_label
from dataset import decode_label

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import os


class Classifier:
    def __init__(self, model):
        self.model = model
        self.hog = cv2.HOGDescriptor()
    
    def compute_difference(self, cv2_board_img, cv2_char_img):
        low_bound = (100, 200)
        mid_bound = (200, 300)
        high_bound = (300, 400)

        cv2_board_canny_wide = cv2.Canny(cv2_board_img, low_bound[0], low_bound[1])
        cv2_board_canny_mid = cv2.Canny(cv2_board_img, mid_bound[0], mid_bound[1])
        cv2_board_canny_tight = cv2.Canny(cv2_board_img, high_bound[0], high_bound[1])

        des_board_original = self.hog.compute(cv2_board_img)
        des_board_canny_wide = self.hog.compute(cv2_board_canny_wide)
        des_board_canny_mid = self.hog.compute(cv2_board_canny_mid)
        des_board_canny_tight = self.hog.compute(cv2_board_canny_tight)

        cv2_char_canny_mid = cv2.Canny(cv2_char_img, mid_bound[0], mid_bound[1])
        
        des_char_original = self.hog.compute(cv2_char_img)
        des_char_canny_mid = self.hog.compute(cv2_char_canny_mid)

        diff_original = np.sqrt(np.abs((des_board_original-des_char_original))).sum()
        diff_canny_wide = np.sqrt(np.abs((des_board_canny_wide-des_char_canny_mid))).sum()
        diff_canny_mid = np.sqrt(np.abs((des_board_canny_mid-des_char_canny_mid))).sum()
        diff_canny_tight = np.sqrt(np.abs((des_board_canny_tight-des_char_canny_mid))).sum()

        diff = min(diff_canny_wide, diff_canny_mid, diff_canny_tight)*0.6 + diff_original*0.4

        return diff

    def pred_category(self, image_path):
        testset = DemoDataset(image_path, transform=transforms.Compose([
                            transforms.Resize((224, 224))]))
        test_dataloader = DataLoader(testset, batch_size=1, shuffle=False)

        for data in test_dataloader:
            images, labels = data
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
            pred_category = label = decode_label(predictions[0])
            print('Predicted category:', pred_category)
            return pred_category

    def predict_char(self, pred_class, folder_path, image_path):
        hog = cv2.HOGDescriptor()
        img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (200, 200), interpolation = cv2.INTER_AREA)
        
        character_files = []
        character_diffs = []
        for _, categories, _ in os.walk(folder_path):
            for category in categories:
                if (category == pred_class):
                    category_path = os.path.join(folder_path, category)
                    for _, _, char_files in os.walk(category_path):
                        for char_file in char_files:
                            if not (char_file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
                                continue

                            temp_file_path = os.path.join(category_path, char_file)

                            temp_img = cv2.resize(cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE), (200, 200), interpolation = cv2.INTER_AREA)
                            temp_flip_img = cv2.flip(temp_img, 1)

                            temp_des = hog.compute(temp_img)
                            temp_flip_des = hog.compute(temp_flip_img)

                            temp_diff = self.compute_difference(img, temp_img)
                            temp_flip_diff = self.compute_difference(img, temp_flip_img)

                            if (temp_diff < temp_flip_diff):
                                character_files.append((temp_file_path, False))
                                character_diffs.append(temp_diff)
                            else:
                                character_files.append((temp_file_path, True)) # We need to flip the character image
                                character_diffs.append(temp_flip_diff)
                    
            idx = character_diffs.index(min(character_diffs))
            result_character_path, filp = character_files[idx]

            result_img = cv2.imread(result_character_path)
            if filp:
                result_img = cv2.flip(result_img, 1)
            
            cv2.imshow("result", cv2.resize(result_img,(result_img.shape[1]//4, result_img.shape[0]//4)))
            cv2.waitKey()