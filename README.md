# PennAICampProject
This project is a little toy work of Group 2 as the final project in 2021's Penn AI Summer Camp, which focuses on Egyptian hieroglyphic recognition from images of boards with the characters. 

To be more specific, our program consists of two parts, namely the segmentation and classification. 
The segmentation part is where characters are extracted from the input image. We intended to deal with the task with SOLOv2. However, the SOLOv2 is trained on the COCO dataset, where only covers a small subset of Egyption hieroglyphics. As a result, the program is limited to merely segment characters like birds. Therefore, we implemented an interactive segmentation program to allow users to draw the AOI.
While in the classification part, which includes two stages, the image is mapped to the corresponding Egypt's character in our Egyptian hieroglyphic library. We did transfer learning on resnet50 with our home-made dataset which categorizes the characters. In the fisrt stage, the trained resnet50 predicts the category. And in the second stage, the input image will be mapped to the one with the least difference among all the characters in the corresponding category.

Finally, I'd like to express my gratitude to my team. Thank TA Charles for leading the project and his efforts in the segmentation part. Thank TA Eric for accelerating the model training process and help on the classification part. Also, thank Kevin for the project idea. Thank Heather for the dataset for SOLOv2. Thank Calvin for some coding stuff and a good presentation. I really had fun in this project.



