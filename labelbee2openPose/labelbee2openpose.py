import json
import numpy as np
import argparse
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


def load_json(filename):
    data = ''

    with open(filename, 'r+') as f:
        data = f.read()

    json_data = json.loads(data)

    return json_data


def save_openPose(out_filename, outData):
    with open(out_filename, 'w+') as f:
        f.write(json.dumps(outData, indent=2))

    print('File is written in ' + out_filename)

    return


def convert_to_openpose(frames_annon, vid_data):
    openPose = OrderedDict()
    openPose['info'] = fill_metadata(vid_data["name"])
    openPose['images'] = list()
    openPose['categories'] = fill_categories()
    openPose['annotations'] = list()

    for frame_id in range(len(frames_annon)):
        if frames_annon[frame_id] is None:
            continue

        frame = frames_annon[frame_id]
        img_desc = OrderedDict()
        img_desc['id'] = frame_id

        # File_name format is the frame_id with at least 12 digits
        img_desc['file_name'] = "0" * (12 - len(str(frame_id)))
        img_desc['file_name'] += str(frame_id) + '.jpeg'

        img_desc['height'] = vid_data["height"]
        img_desc['width'] = vid_data["width"]

        openPose['images'].append(img_desc)

        id_count = 1

        for bee_id in frame.keys():
            bee_annon = frame[bee_id]
            annon = OrderedDict()
            annon['id'] = int(bee_id) + id_count
            id_count += 1

            annon["iscrowd"] = 0
            annon["image_id"] = frame_id
            annon["category_id"] = 1
            annon["num_keypoints"] = 5

            annon["segmentation"] = get_box_coords(bee_annon)
            annon["bbox"] = annon["segmentation"]
            annon["area"] = bee_annon["width"] * bee_annon["height"]
            annon["keypoints"] = extract_parts(bee_annon["parts"])

            openPose['annotations'].append(annon)

    return openPose


# Do flexible whit more parts
PARTS = dict()
PARTS["tail"] = 0
PARTS["abdomen"] = 0
PARTS["head"] = 2
PARTS["thorax"] = 4
PARTS["torax"] = 4
PARTS["rant"] = 6
PARTS["lant"] = 8


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description="Convert from labelbee format to OpenPose format")

    parser.add_argument('-if', '--infile',
                        required=True, help="Labelbee Format file")

    parser.add_argument('-H', '--height', type=int,
                        default=2560, help="Height of the video")

    parser.add_argument('-W', '--width', type=int,
                        default=1440, help="Width of the video")

    parser.add_argument('-of', '--outfile',
                        default="output.json",
                        help="Output file (OpenPose format)")

    args = parser.parse_args()

    filename = args.infile
    output_filename = args.outfile

    # Fill video metadata
    videoData = dict()
    videoData["name"] = get_video_name(filename)
    videoData["height"] = args.height
    videoData["width"] = args.width

    # Load Labelbee format
    lbee_data = load_json(filename)

    # Convert to Open Pose format
    openPoseData = convert_to_openpose(lbee_data, videoData)

    # Save in output file
    save_openPose(output_filename, openPoseData)
