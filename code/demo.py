from segmentation import select_char
from classify import Classifier
from net import MyNet

import argparse
import torch
import os

if __name__ == '__main__':

    # disable warning
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    parser = argparse.ArgumentParser(description='Select a region from a board image and recognize its character.')
    parser.add_argument('-board_path', help='Path of your board image', default='./data/board/board1.jpg')
    parser.add_argument('-bbox_path', help='Path to save the bounded region', default='./test.jpg')
    args = parser.parse_args()

    board_path = args.board_path
    test_img_path = './test.jpg'
    select_char(board_path, test_img_path)

    model_path = './models/model_on_boardset.pth'
    model = MyNet(category_num=9)
    model.load_state_dict(torch.load(model_path))

    char_folder_path = './data/categorized_characters'
    classifier = Classifier(model)

    pred_category = classifier.pred_category(test_img_path) # class
    classifier.predict_char(pred_category, char_folder_path, test_img_path)


