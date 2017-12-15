
import sys
import json
import numpy as np
from math import cos, sin, radians
from collections import OrderedDict


def get_video_name(json_name):
    json_name = json_name.split('-Tracks')
    return json_name[0]


def fill_categories():
    cat = OrderedDict()
    cat['id'] = 1
    cat['skeleton'] = [[1, 3], [3, 2], [2, 4], [2, 5]]
    cat['keypoints'] = ['Tail', 'Head', 'Torax', 'RAnt', 'LAnt']
    cat['super_category'] = "bee"
    cat['name'] = "bee"
    return [cat]


def fill_metadata(video_name):
    md = OrderedDict()
    md['description'] = video_name
    md['video'] = video_name
    md['Date_created'] = "dic_2017"
    md['year'] = 2017
    md['contributor'] = "I. Rodriguez, J. Chan & R. Megret"
    md['version'] = "0.01"
    return md


def get_rotated_pt(x, y, theta, center):
    cos_theta = cos(radians(theta))
    sin_theta = sin(radians(theta))

    n_x = (x - center[0]) * cos_theta - (y - center[1]) * sin_theta + center[0]
    n_y = (x - center[0]) * sin_theta + (y - center[1]) * cos_theta + center[1]

    return [n_x, n_y]


# def get_box_coords(x, y, height, width, theta):
#     center = [x + width/2, y + height/2]
#     pts = get_rotated_pt(x, y, theta, center)
#     pts += get_rotated_pt(x, y + height, theta, center)
#     pts += get_rotated_pt(x + width, y + height, theta, center)
#     pts += get_rotated_pt(x + width, y, theta, center)
#     return [pts]

def get_box_coords(bee_annotation):
    x = bee_annotation["x"]
    y = bee_annotation["y"]
    width = bee_annotation["width"]
    height = bee_annotation["height"]
    theta = bee_annotation["angle"]

    center = [x + width/2, y + height/2]
    pts = get_rotated_pt(x, y, theta, center)
    pts += get_rotated_pt(x, y + height, theta, center)
    pts += get_rotated_pt(x + width, y + height, theta, center)
    pts += get_rotated_pt(x + width, y, theta, center)

    return [pts]


def extract_parts(parts_list):
    keypoints = np.zeros(10).tolist()

    for part in parts_list:
        p_name = str(part["label"]).lower()
        index = PARTS[p_name]
        keypoints[index] = part["posFrame"]["x"]
        keypoints[index + 1] = part["posFrame"]["y"]

    return keypoints


# Do flexible whit more parts
PARTS = dict()
PARTS["tail"] = 0
PARTS["abdomen"] = 0
PARTS["head"] = 2
PARTS["thorax"] = 4
PARTS["torax"] = 4
PARTS["rant"] = 6
PARTS["lant"] = 8

if len(sys.argv) == 3:
    filename = sys.argv[1]
    output_filename = sys.argv[2]
elif(len(sys.argv) == 2):
    filename = sys.argv[1]
    output_filename = 'output.json'
else:
    print("To run:\n")
    print("\t\tpython " + sys.argv[0] + " <labelbee json> <output file>\n")
    exit()


VIDEO_NAME = get_video_name(filename)
VIDEO_HEIGHT = 2560
VIDEO_WIDTH = 1440

OPEN_POSE = OrderedDict()
OPEN_POSE['images'] = list()
OPEN_POSE['annotations'] = list()
OPEN_POSE['categories'] = fill_categories()
OPEN_POSE['info'] = fill_metadata(VIDEO_NAME)


data = ''
with open(filename, 'r+') as f:
    data = f.read()

lbee = json.loads(data)


for i in range(len(lbee)):
    if lbee[i] is None:
        continue

    frame = lbee[i]
    img_desc = OrderedDict()
    img_desc['id'] = i
    img_desc['file_name'] = "0" * (12 - len(str(i))) + str(i) + '.jpeg'
    img_desc['height'] = VIDEO_HEIGHT
    img_desc['width'] = VIDEO_WIDTH
    OPEN_POSE['images'].append(img_desc)
    id_count = 1
    for bee_id in frame.keys():
        bee_annon = frame[bee_id]
        annon = OrderedDict()
        annon['id'] = int(bee_id) + id_count
        id_count += 1
        annon["iscrowd"] = 0
        annon["image_id"] = i
        annon["category_id"] = 1
        annon["num_keypoints"] = 5

        annon["segmentation"] = get_box_coords(bee_annon)
        annon["bbox"] = annon["segmentation"]
        annon["area"] = bee_annon["width"] * bee_annon["height"]
        annon["keypoints"] = extract_parts(bee_annon["parts"])

        OPEN_POSE['annotations'].append(annon)


with open(output_filename, 'w+') as f:
    f.write(json.dumps(OPEN_POSE, indent=2))

print('File is written in ' + output_filename)
