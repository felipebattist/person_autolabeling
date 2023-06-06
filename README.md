# person_autolabeling
"Person AutoLabeling" automates object detection data labeling using YOLO models. It reads YOLO format data, generates labeled bounding boxes for objects in images, and assigns accurate labels. Simplify your labeling process with this convenient tool powered by YOLO's object detection capabilities.

Install necessary libraries:

```
$ pip install -r requirements.txt
```

How to use:

In the code change the file folder_path variable to the location of your image folder, than:

```
$ python label.py
```

The following commands can be used:

*    mouse_click = create a bounding box
*    y = call YOLO predictions
*    , or . = to select a box
*    e = erase the selected box
*    r = Remove all boxes from the image
*    z = go to next image 
*    p = positive fine-tuning
*    n = negative fine-tuning
*    w a s d = fine-tuning control
*    esq = quit
*    q = quit without saving

