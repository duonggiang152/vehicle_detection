## About project: Vehicle_Detection

# Project aim to learn about rest net. Use Base model is resnetfpn which are designed by tensorflow <a>https://github.com/tensorflow/models</a> 

Data was collected by two kaggle dataset 



dataset 1: <a>https://www.kaggle.com/datasets/sshikamaru/car-object-detection</a>


dataset 2: <a>https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset/data</a>


# trained model: https://drive.google.com/file/d/1Aqk769C6iIBpKZWGYcD-_O1k5ajxQxGa/view?usp=sharing


Both datesets are needed preprocessing to feed tensorflow model.

csv_to_coco_anotation.py provide a script to convert first dataset annotation to coco format annotation

yolo_to_coco_anotation.py provide a script to convert second dataset annotation to coco format annotation


CLIs folder hold script for convert coco format annotation to TFRecord which are required for tensorflow prebuilt model

object_detection_built note book contains model detail, way to train model

tracking_car note book contains way to use object detection model as part of DEEP SORT tracking algorithm
![image](https://github.com/user-attachments/assets/88bcc5d1-3c99-4619-b02f-56eaa1ef7087)
