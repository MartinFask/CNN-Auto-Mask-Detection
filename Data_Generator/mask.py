"""
This source code is created by Prajna Bhandary
# find the original code at:
# https://github.com/prajnasb/observations/blob/master/mask_classifier/Data_Generator/mask.py
"""


import os
import numpy as np
from PIL import Image, ImageFile

__version__ = '0.3.0'


def create_mask(image_path, folder_path, mask_path=None):
    pic_path = image_path
    if mask_path is None:
        mask_path = "./masks/1.png"
    show = False
    FaceMasker(pic_path, mask_path, folder_path, show, "hog").mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, dest_folder='', show=False, model='hog'):
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.dest_folder = dest_folder

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            if self.show:
                self._face_img.show()

            # save
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.6 # Note: here
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v) * 2)  # Note: here

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2 + new_height//12 # Note: here

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        drive, path_and_file = os.path.splitdrive(self.face_path)
        path, file = os.path.split(path_and_file)
        path_splits = os.path.splitext(file)
        new_face_path = path + self.dest_folder + path_splits[0] + '-with-coffee' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    faces_path = './faces/5/'
    counter = 0
    for img in os.listdir(faces_path):
        im_path = os.path.join(faces_path, img)
        if not os.path.isfile(im_path):
            continue
        mask_path = "./objects/hand2.png"
        print("mask path {}".format(mask_path))
        create_mask(image_path=im_path, folder_path="/added/", mask_path=mask_path)
        # counter += 1
        # if counter > 5:
        #     break


# Location adjustment for different objects:

# for coffeecup and coffeecup2 :
# width_ratio = 1.8
# new_height *= 1.8
# center_y += new_height//4

# for hand :
# width_ratio = 1.5
# new_height *= 1.8
# center_y += new_height//7

# for glove :
# width_ratio = 1.9
# new_height *= 2.2
# center_y += new_height//7

# for hand 2:
# width_ratio = 1.6
# new_height *= 2
# center_y += new_height//12

# for glove 2:
# width_ratio = 1
# new_height *= 1.8
# center_y += new_height//5

# for pen:
# width_ratio = 1
# new_height *= 1
# center_y += new_height//10 * [-1, 1][np.random.choice(2)]

# for masks:
# width_ratio = 2
# new_height *= 2.1
# center_y += 10