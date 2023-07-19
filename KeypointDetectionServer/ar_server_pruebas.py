#!/usr/bin/env python3
"""
Image matching server
 - Keypoint detection with ASLFeat
 - (+) 3D Model projection to a recognized surface
"""
import yaml
import cv2
import numpy as np
from objloader_simple import *
import math

from models import get_model
import glob

import flask
import os
import uuid
app = flask.Flask(__name__)

##############################################

IMAGE_FOLDER = './saved_imgs'
model_path = './model.ckpt-0'
OUTPUT_FOLDER = './processed_imgs'

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10
DEFAULT_COLOR = (237, 36, 75) # BGR for OpenCV

##############################################


@app.route('/process/', methods=['POST'])
def process_images():

    # Remove saved images so folder does NOT get full
    savedimgs = glob.glob('./saved_imgs/*')
    for f in savedimgs:
        try:
            os.remove(f)
        except Exception as e:
            print(f'Failed to remove file {f} - Exception: {e}')
    
    # Remove processed images so folder does NOT get full
    processedimgs = glob.glob('./processed_imgs/*')
    for f2 in processedimgs:
        try:
            os.remove(f2)
        except Exception as e:
            print(f'Failed to remove file {f2} - Exception: {e}')
    

    # Check if an image file was uploaded
    if 'image' not in flask.request.files:
        print('No image file found')

    image_file = flask.request.files['image']
    print("\nImage received? ", image_file != None)

    image_uuid = str(uuid.uuid4())
    output_name = f'{image_uuid}.jpg'
    print(f'Processing image {image_uuid}')

    image_path = os.path.join(IMAGE_FOLDER, output_name)
    image_file.save(image_path)

    homography = None

    # Read the image (frame)
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    # Find and draw the keypoints of the frame
    des_frame, kp_frame, _ = model.run_test_data(frame_gray)

    # Transform to OpenCV keypoints
    cv_kpts_frame = [cv2.KeyPoint(kp_frame[i][0], kp_frame[i][1], 1)
                        for i in range(kp_frame.shape[0])]

    frame_with_keypoints = cv2.drawKeypoints(frame, cv_kpts_frame, 0, (0, 255, 0))
    cv2.imwrite('frame_with_kpts.jpg', frame_with_keypoints)
    
    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)

    # Sort them by distance. The lower the distance, the better
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute Homography if enough matches are found
    if len(matches) > MIN_MATCHES:
        # Differentiate between source points and destination points
        src_pts = np.float32([cv_kpts_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([cv_kpts_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Draw a rectangle that marks the found model in the frame
        h, w , _ = model_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines  
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


        # If a valid homography matrix was found, render the object on the image
        if homography is not None:
            try:
                # Obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)

                # Render the object on the image
                frame = render(frame_with_keypoints, obj, projection, model_img, False)
            except:
                pass
        
        # Draw matches. It will display both images and link keypoints from one image to another
        # frame = cv2.drawMatches(model_img, cv_kpts_model, frame, cv_kpts_frame, matches[:10], 0, flags=2)

    
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    cv2.imwrite(output_path, frame)

    return flask.send_file(output_path, mimetype='image/jpeg')



##############################################
"""
Auxiliary functions
"""

def render(img, obj, projection, model_img, color=False):
    """
    Render a loaded obj model into the current image frame
    """
    print('Object is being rendered')
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.1
    h, w, _ = model_img.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


##############################################


if __name__ == '__main__':

    with open('configs/dibujar_kpts.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = get_model('feat_model')(model_path, **config['net'])

    # Keep this always loaded. It wont change
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # Load the reference surface that will be searched in the image
    model_path = 'tfg_model.jpg'
    model_img = cv2.imread(model_path, cv2.IMREAD_UNCHANGED) # OpenCV lee como BGR
    model_gray = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    des_model, kp_model, _ = model.run_test_data(model_gray)

    cv_kpts_model = [cv2.KeyPoint(kp_model[i][0], kp_model[i][1], 1)
                        for i in range(kp_model.shape[0])]
    
    
    model_with_keypoints = cv2.drawKeypoints(model_img, cv_kpts_model, 0, (0, 255, 0))
    cv2.imwrite('model_with_kpts.jpg', model_with_keypoints)

    # Load 3D model from OBJ file
    obj = OBJ('AR_models\cow.obj', swapyz=True)


    app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000))