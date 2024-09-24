# Graduate-project

#The project is called :"Label-based Region Recoloring in Clothing Images via YOLOv8 Semantic Segmentation Model Optimization"
This project is mainly based on the semantic segmentation of different regions of fashion dataset pictures on the basis of YOLOv8 model, and then custom color change is performed on the segmented part.
The application scenario is: fashion shopping website scenario of e-commerce platform. This technology can reduce the processing flow of pictures and improve work efficiency for merchants before selling clothing products.
At the same time, this technology provides more color matching effects for users who buy clothing products on the online platform, and users can also customize the regional color change according to their desired color.

#The datasets used in this project are public datasets: the website can be found at https://github.com/bearpaw/clothing-co-parsing

# About the code file for this project:
At present, due to the large size of the data set file and other experimental files, and the github website has limits on the size of uploaded files, it is impossible to upload all the files to github.
I created a new public link on the Google Drive website: https://drive.google.com/drive/folders/15RYiHDzY36sqTB8ooMm-CjO0dWdJQGgB?usp=sharing
This link contains all the code files, training results, and dataset files for the research project. (Publicly visible to all)

# About the code folder structure:
Firstly, according to the project structure and content, this project is divided into four parts, which are:
Part1-Dataset preprocessing, Part2- Model training files, Part3-Experiment file, Part4-change color
Let's take a closer look at what files are in each of the four folders and what they do.

# part1-Dataset preprocessing:
1.Dataset preprocessing.py (This file preprocesses the dataset file, mainly converting the mat format file into a txt file)
2.NewClothing folder (this folder is the new dataset file obtained by preprocessing the dataset)
3.dataset folder (This folder is mainly divided according to the dataset of the NewClothing folder, and changed into a format that can be recognized by YOLOv8 and used for training)

# Part2- Model training files:
1. newCloth. yaml file (this file is the tag name file of the dataset for this project, modified according to the format of YOLOv8.yaml)
2. YOLOv8_file folder (most of the content in this folder downloaded from YOLOv8 official website, link: https://github.com/ultralytics/ultralytics/tree/main/ultralytics).
But in this file, the main files used are "train.py", "YOLOv8-bifpn.yaml", "YOLOv8-p6.yaml", "YOLOv8n-seg_CBAM.yaml".
These yaml files are what will be loaded into "train.py" for training
This folder also contains the interface files "tasks.py", "block.py", "_init_.py ", etc

# Part3-Experiment file:
In this file mainly contains all the files about the experimental part done by the project, where each model and its training results are stored in a separate folder.
There are seven folders in total. Include:
"YOLOv8-p6", "YOLOv8n-seg+BIFPN", "YOLOv8n-seg+CBAM", "YOLOv8n-seg(load yolov8n.pt file)", "YOLOv8n-seg(Don't load yolov8n.pt file) ", "YOLOv8x-seg(load yolov8n.pt file)", "delete label name"

# Part4-change color:
This file contains the code file for how to change the color of the image: "change_color_code.py"
change color result folder: "Change color Result", which contains the image effect after changing the color of different areas of the image.

# About LLM
In this project, Large Language Model (LLM) was used to make changes and adjustments when code errors and formatting problems occurred.



