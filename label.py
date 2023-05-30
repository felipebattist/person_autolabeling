import cv2
import csv
import glob

import yoloV8
from yoloV8 import *

# Global variables
img = None
tl_list = []
br_list = []
object_list = []
current_tl = []
current_br = []
current_object = ""
selected_box_index = -1
current_mod_state = 1

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global img, tl_list, br_list, object_list, current_tl, current_br, current_object, selected_box_index

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_box_index = -1

        for i in range(len(tl_list)):
            if tl_list[i][0][0] <= x <= br_list[i][0][0] and tl_list[i][0][1] <= y <= br_list[i][0][1]:
                selected_box_index = i
                break

        if selected_box_index == -1:
            current_tl = [(x, y)]
            current_br = [(x, y)]  # Assign the same coordinates initially

    elif event == cv2.EVENT_LBUTTONUP:
        current_br = [(x, y)]

        if current_tl != current_br:  # Only add non-zero size bounding boxes
            current_tl, current_br = swap_coordinates(current_tl[0], current_br[0])  # Swap coordinates if necessary
            current_object = ""  # Set your desired object label here

            if selected_box_index == -1:
                tl_list.append(current_tl)
                br_list.append(current_br)
                object_list.append(current_object)
            else:
                tl_list[selected_box_index] = current_tl
                br_list[selected_box_index] = current_br

        img = img_copy.copy()
        for i in range(len(tl_list)):
            cv2.rectangle(img, tl_list[i][0], br_list[i][0], (0, 255, 0), 2)
        if selected_box_index != -1:
            cv2.rectangle(img, tl_list[selected_box_index][0], br_list[selected_box_index][0], (255, 0, 0), 2)
        cv2.imshow("Image", img)

def swap_coordinates(tl, br):
    """Swap the coordinates if necessary to ensure consistent bounding box format"""
    if tl[0] > br[0]:
        tl, br = br, tl
    if tl[1] > br[1]:
        tl, br = (tl[0], br[1]), (br[0], tl[1])
    return [(tl[0], tl[1])], [(br[0], br[1])]



def load_bounding_boxes_from_csv(csv_file_path):
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            image_path, tl, br, obj = row
            tl = eval(tl)
            br = eval(br)
            tl_list.append(tl)
            br_list.append(br)
            object_list.append(obj)


def delete_selected_box():
    global selected_box_index, img

    if selected_box_index != -1:
        tl_list.pop(selected_box_index)
        br_list.pop(selected_box_index)
        object_list.pop(selected_box_index)
        selected_box_index = -1

        img = img_copy.copy()
        for i in range(len(tl_list)):
            cv2.rectangle(img, tl_list[i][0], br_list[i][0], (0, 255, 0), 2)
        cv2.imshow("Image", img)


def change_selected_box(key):
    global selected_box_index, img

    if key == ord("."):  # Press '.' key to move up
        selected_box_index = max(selected_box_index - 1, 0)
    elif key == ord(","):  # Press ',' key to move down
        selected_box_index = min(selected_box_index + 1, len(tl_list) - 1)

    img = img_copy.copy()
    for i in range(len(tl_list)):
        cv2.rectangle(img, tl_list[i][0], br_list[i][0], (0, 255, 0), 2)
    if selected_box_index != -1:
        cv2.rectangle(img, tl_list[selected_box_index][0], br_list[selected_box_index][0], (255, 0, 0), 2)
    cv2.imshow("Image", img)

def increase_top():
    global selected_box_index, tl_list
    if selected_box_index != -1:
        tl_list[selected_box_index][0] = (tl_list[selected_box_index][0][0], tl_list[selected_box_index][0][1] - 1)

def decrease_top():
    global selected_box_index, tl_list
    if selected_box_index != -1:
        tl_list[selected_box_index][0] = (tl_list[selected_box_index][0][0], tl_list[selected_box_index][0][1] + 1)

def increase_bottom():
    global selected_box_index, br_list
    if selected_box_index != -1:
        br_list[selected_box_index][0] = (br_list[selected_box_index][0][0], br_list[selected_box_index][0][1] + 1)

def decrease_bottom():
    global selected_box_index, br_list
    if selected_box_index != -1:
        br_list[selected_box_index][0] = (br_list[selected_box_index][0][0], br_list[selected_box_index][0][1] - 1)

def increase_right():
    global selected_box_index, br_list
    if selected_box_index != -1:
        br_list[selected_box_index][0] = (br_list[selected_box_index][0][0] + 1, br_list[selected_box_index][0][1])

def decrease_right():
    global selected_box_index, br_list
    if selected_box_index != -1:
        br_list[selected_box_index][0] = (br_list[selected_box_index][0][0] - 1, br_list[selected_box_index][0][1])

def increase_left():
    global selected_box_index, tl_list
    if selected_box_index != -1:
        tl_list[selected_box_index][0] = (tl_list[selected_box_index][0][0] - 1, tl_list[selected_box_index][0][1])

def decrease_left():
    global selected_box_index, tl_list
    if selected_box_index != -1:
        tl_list[selected_box_index][0] = (tl_list[selected_box_index][0][0] + 1, tl_list[selected_box_index][0][1])

# Read images from a folder
folder_path = "images"  # Replace with the actual folder path
image_files = glob.glob(folder_path + "/*.jpg")  # Modify the extension if needed

# Read existing bounding boxes from the CSV file
existing_images = set()
with open("bounding_boxes.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        existing_images.add(row["File Path"])

# Create a CSV file and write header
csv_file = open("bounding_boxes.csv", "a", newline="")
csv_writer = csv.writer(csv_file)

# Iterate through each image
for image_file in image_files:
    if image_file in existing_images:
        print(f"Skipping {image_file} (already processed)")
        continue

    count = 0
    img = cv2.imread(image_file)
    img_copy = img.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.imshow("Image", img)

    # Reset variables for each image
    tl_list = []
    br_list = []
    object_list = []
    selected_box_index = -1

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):  # Press 'y' key to load bounding boxes from CSV
            yoloV8.create_yolo_boxes(image_file)
            csv_file_path = "yolo_boxes.csv"
            load_bounding_boxes_from_csv(csv_file_path)
            img = img_copy.copy()
            for i in range(len(tl_list)):
                cv2.rectangle(img, tl_list[i][0], br_list[i][0], (0, 255, 0), 2)
            cv2.imshow("Image", img)

        if key == 27:  # Press 'Esc' key to exit
            csv_file.close()
            cv2.destroyAllWindows()
            exit()

        if key == ord("e"):  # Press 'e' key to erase the selected box
            delete_selected_box()
        elif key == ord(".") or key == ord(","):  # Press '.' or ',' key to change the selected box
            change_selected_box(key)
        elif key == ord("r"):  # Press 'r' key to reset bounding boxes
            img = img_copy.copy()
            tl_list = []
            br_list = []
            object_list = []
            selected_box_index = -1
            cv2.imshow("Image", img)
        elif key == ord("n"): # Press 'n' to activate negative fine tunning
            current_mod_state = 0
        elif key == ord("p"): # Press 'p' to activate positive fine tunning
            current_mod_state = 1
        elif key == ord("w"):  # Press 'w' key to increase/decrease top coordinate
            if current_mod_state:
                increase_top()
            elif not current_mod_state: 
                decrease_top()
            change_selected_box(key)
        elif key == ord("s"):  # Press 's' key to increase/decrease bottom coordinate
            if current_mod_state:
                increase_bottom()
            elif not current_mod_state:
                decrease_bottom()
            change_selected_box(key)
        elif key == ord("a"):  # Press 'a' key to increase/decrease left coordinate
            if current_mod_state:
                increase_left()
            elif not current_mod_state:
                decrease_left()
            change_selected_box(key)
        elif key == ord("d"):  # Press 'd' key to increase/decrease right coordinate
            if current_mod_state:
                increase_right()
            elif not current_mod_state:
                decrease_right()
            change_selected_box(key)
        elif key == ord("z"):  # Press 'z' key to save bounding boxes and move to the next image
            csv_rows = zip([image_file] * len(tl_list), tl_list, br_list, object_list)
            csv_writer.writerows(csv_rows)
            csv_file.flush()
            break
        elif key == ord("q"):  # Press 'q' key to quit without saving
            csv_file.close()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

csv_file.close()
print("Bounding box data saved in bounding_boxes.csv")
