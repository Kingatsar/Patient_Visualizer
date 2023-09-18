""" test on aruco marker """

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json


def calculate_distance(aruco_x, aruco_y, center_x=320, center_y=240) -> float:
    """
        Calculate the distance between the top left corner of the aruco position and the center of the image display
    :param aruco_x: x value of the top left corner of the aruco
    :type aruco_x: float
    :param aruco_y: y value of the top left corner of the aruco
    :type aruco_x: float
    :param center_x: x value of the center of the image display
    :type center_x: int
    :param center_y: y value of the center of the image display
    :type center_y: int
    :return: distance between the top left corner of the aruco and the center
    :rtype: float
    """
    x_diff = np.abs(aruco_x - center_x)
    y_diff = np.abs(aruco_y - center_y)
    return np.sqrt(x_diff ** 2 + y_diff ** 2)


def draw_patient_info(dict_patient):
    """
        Creates an image and draws the patient's information on it
    :param dict_patient: dictionary containing the information of the patient
    :type dict_patient: dictionary
    :return: image where the information of the patient is written
    :rtype: numpy array

    https://gist.github.com/imneonizer/b64cdd8e2dc23451f5d8caf8279b3ff5 to write text on multiple lines

    """
    # get patient's info
    patient0 = dict_patient

    # parameters for the text
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontSize = 0.5
    fontThickness = 1
    max_width = 0  # width
    height_max = 0

    # calculating the height and width for the new image
    for val in patient0.keys():
        a1 = f'{val} : {patient0[val]}'
        textSize = cv2.getTextSize(a1, fontFace, fontSize, fontThickness)[0]
        a, b = textSize[0], textSize[1]
        height_max += 2 * b
        if a > max_width:
            max_width += a

    # create new image
    image = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

    # draw text in new image
    org_x, org_y = 0, 0
    for val in patient0.keys():
        a = f'{val} : {patient0[val]}'
        textSize = cv2.getTextSize(a, fontFace, fontSize, fontThickness)
        textSize = textSize[0]
        step = textSize[1] + 5
        org_y += step
        cv2.putText(image, a, (org_x, org_y), fontFace, fontSize, (255, 0, 255), fontThickness,
                    lineType=cv2.LINE_AA)

    # only keep the written part of the image
    final = extractFace(image, 0, 0, max_width, height_max)

    return final


def extractFace(imgFace, x, y, w, h):
    """
        Extracts (crops) the face part of the image to warp it
    :param imgFace: the part where the face is covered
    :type imgFace: numpy array
    :param x: origin of imgFace in the x-axis
    :type x: int
    :param y: origin of imgFace in the y-axis
    :type y: int
    :param w: width of imgFace
    :type w: int
    :param h: height of imgFace
    :type h: int
    :return: image of the extracted face
    """
    # create points
    pts1 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    # transform matrix
    trans_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # warp the image
    dim = (w, h)
    imgWarped = cv2.warpPerspective(imgFace, trans_matrix, dim)

    return imgWarped


def detectFace(img) -> list:
    """
        Detects if there is a face or not
    :param img: image in which we want to detect a face
    :type img: numpy array
    :return: a list containing a boolean stating if there are faces or not and the list of faces or None
    :rtype: list
    """
    # convert images to gray scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face detection
    face_cascade = cv2.CascadeClassifier('ressources/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)

    if len(faces) != 0:
        return [True, faces]
    else:
        return [False, None]


def face_recognition_compare(img, img_patient_face, drawFace=True) -> list:
    """
        Detects a face and compares the face in the image with the face from the patient's data.
    :param img: image on which we want to detect a face
    :type img: numpy array
    :param img_patient_face: face of the patient in the database
    :type img_patient_face: numpy array
    :param drawFace: if we want to draw a rectangle on the face or not
    :return: a list containing the image, the mean squared error, the x and y value of the top left rectangle
    :rtype: list
    """
    # -------- detect faces -------- #
    detected, faces = detectFace(img)[0], detectFace(img)[1]

    # draws a rectangle on the face
    # suppose that we would only detect one face
    if drawFace:
        ifx, ify, ifw, ifh = 0, 0, 0, 0
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgFace = extractFace(img, x, y, w, h)
            ifx, ify, ifw, ifh = x, y, w, h

    # -------- Compare the 2 images -------- #
    # change the dimension of the patient's face in the database to fit the one from the camera
    iPFh, iPFw = img_patient_face.shape[0], img_patient_face.shape[1]
    pts1 = np.array([[ifx, ify], [ifx + ifw, ify], [ifx + ifw, ify + ifh], [ifx, ify + ifh]])
    pts2 = np.array([[0, 0], [iPFw, 0], [iPFw, iPFh], [0, iPFh]])
    trans_mat, _ = cv2.findHomography(pts2, pts1)
    dim = (ifw, ifh)
    warpedIPF = cv2.warpPerspective(img_patient_face, trans_mat, dim)

    # Calculate the mean squared error
    # normalize images
    mat_ones = np.ones((ifw, ifh, 3))
    imgF_norm = np.linalg.norm(imgFace + mat_ones)
    imgPF_norm = np.linalg.norm(warpedIPF + mat_ones)
    imgF_normalized = imgFace / imgF_norm
    imgPF_normalized = warpedIPF / imgPF_norm
    # calculate the mean squared erro
    squared_diff = np.abs(imgPF_normalized - imgF_normalized) ** 2
    mse = np.nanmean(squared_diff)

    return [img, mse, ifx, ify]


def loadPatientInfo(path) -> dict:
    """
        Loads the images in the path given (ideally a folder) and links this images to the patient's info
    :param path: path of the images' folder
    :type path: string
    :return: returns a dictionary where for a given marker id, a patient's image and info are given
    :rtype: dict
    """
    # -------- patient's information retrieved -------- #
    # Opening JSON file
    f = open('database.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    # close JSON file
    f.close()
    patient0 = data["0"]["information"]
    patient1 = data["1"]["information"]
    patients_info = [patient0, patient1]

    # give the complete  list of images
    myList = os.listdir(path)

    # -------- create a dictionary -------- #
    imgAug_dict = {}

    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        # store image in the dictionary for a given key
        imgAug_dict[key] = [imgAug, patients_info[key]]

    return imgAug_dict


def DetectArucoMarkers(img, markerSize: int = 4, totalMarkers: int = 250, draw=True) -> list:
    """
        Find aruco markers on a given image
    :param img: image from which we want to find aruco marker
    :param markerSize: the size of the aruco
    :type markerSize: int
    :param totalMarkers: total number of aruco possibilities
    :type totalMarkers: int
    :param draw: if we want to draw the aruco
    :type: boolean
    :return: list containing a list of the bounding boxes (or corners) and  the ids of the aruco markers
    :rtype: list
    """
    # change image to gray
    img_To_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define which dictionary to get from marker
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    aruco_Dictionary = aruco.Dictionary_get(key)
    # create parameters
    aruco_Parameters = aruco.DetectorParameters_create()
    bounding_boxes, ids, rejected_markers = aruco.detectMarkers(img_To_Gray, aruco_Dictionary,
                                                                parameters=aruco_Parameters)
    # -------- draw the markers -------- #
    if draw:
        aruco.drawDetectedMarkers(img, bounding_boxes)

    return [bounding_boxes, ids]


def arucoAugmentImage(corner, imgToDrawOn, patient_detail):
    """
        Augments an image on the aruco marker
    :param corner: the corners of the aruco marker
    :param imgToDrawOn: the image where we want to draw on
    :type imgToDrawOn: numpy array
    :param patient_detail: list containing the patient's image and its details that are in a dictionary
    :param patient_detail: list
    :return: returns the overlay of the image we want to draw on and the augmented image
    :rtype: numpy array
    """
    # the image to augment
    imgToAugment = patient_detail[0]

    # four corner points of the aruco marker
    top_left = corner[0][0][0], corner[0][0][1]
    top_right = corner[0][1][0], corner[0][1][1]
    bottom_right = corner[0][2][0], corner[0][2][1]
    bottom_left = corner[0][3][0], corner[0][3][1]

    # get size of the image we want to augment
    height, width, channels = imgToAugment.shape

    # -------- warp augmented image onto the original image -------- #
    # find homography
    w = np.abs(top_right[0] - top_left[0])
    h = np.abs(bottom_left[1] - top_left[1])
    # position for the patient's image
    pos_top_left = [top_left[0] - w, top_left[1] - 2 * h]
    pos_top_right = [top_right[0] - w, top_right[1] - 2 * h]
    pos_bottom_left = [bottom_left[0] - w, bottom_left[1] - 2 * h]
    pos_bottom_right = [bottom_right[0] - w, bottom_right[1] - 2 * h]
    position1 = np.array([pos_top_left, pos_top_right, pos_bottom_right, pos_bottom_left])
    # creating points
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    homo_matrix, _ = cv2.findHomography(pts2, position1)
    # warping
    dim = (imgToDrawOn.shape[1], imgToDrawOn.shape[0])
    imgOut = cv2.warpPerspective(imgToAugment, homo_matrix, dim)
    # put the imgOut in imgToDrawOn
    for i in range(imgToDrawOn.shape[0]):
        for j in range(imgToDrawOn.shape[1]):
            if not ((imgOut[i, j, 0] == 0) and (imgOut[i, j, 1] == 0) and (imgOut[i, j, 2] == 0)):
                imgToDrawOn[i, j, :] = imgOut[i, j, :]

    # -------- warp patient information on original image -------- #
    # draw the patient's info
    dict_patient = patient_detail[1]
    drawPatient = draw_patient_info(dict_patient)
    dPh, dPw, dPc = drawPatient.shape
    # creating points
    pos_top_left1 = [top_left[0] + w, top_left[1] - 2 * h]
    pos_top_right1 = [top_right[0] + 2 * w + dPw, top_right[1] - 2 * h]
    pos_bottom_left1 = [bottom_left[0] + w, bottom_left[1] - 3 * h + dPh]
    pos_bottom_right1 = [bottom_right[0] + 2 * w + dPw, bottom_right[1] - 3 * h + dPh]
    position2 = np.array([pos_top_left1, pos_top_right1, pos_bottom_right1, pos_bottom_left1])
    pts3 = np.float32([[0, 0], [dPw, 0], [dPw, dPh], [0, dPh]])
    # find homography
    homo_matrix2, _ = cv2.findHomography(pts3, position2)
    # warping
    imgOut2 = cv2.warpPerspective(drawPatient, homo_matrix2, dim)

    # imgToDrawOn[imgOut2 != 0] = imgOut2[imgOut2 != 0]
    for i in range(imgToDrawOn.shape[0]):
        for j in range(imgToDrawOn.shape[1]):
            if not ((imgOut2[i, j, 0] == 0) and (imgOut2[i, j, 1] == 0) and (imgOut2[i, j, 2] == 0)):
                imgToDrawOn[i, j, :] = imgOut2[i, j, :]

    # -------- overlay images -------- #
    result = imgToDrawOn

    return result


def main():
    # define camera
    cap = cv2.VideoCapture(0)
    # load images to augment
    augDic = loadPatientInfo("marker")
    aruco_id = 0
    while True:
        # check success + get image
        success, img = cap.read()

        # ------ Aruco ------ #
        arucoFound = DetectArucoMarkers(img)
        # loop through all the markers and augment each one
        if len(arucoFound[0]) != 0:
            closest_center_image = [99999999, None, None]
            if len(arucoFound[0]) == 1:
                closest_center_image = [closest_center_image[0], arucoFound[0], arucoFound[1]]
            else:
                d = closest_center_image[0]
                for corners, id in zip(arucoFound[0], arucoFound[1]):
                    d1 = calculate_distance(corners[0][0][0], corners[0][0][1])
                    if d1 < d:
                        closest_center_image = [d1, corners, id]
                        d = d1

            if int(closest_center_image[2]) in augDic.keys():
                aruco_corners = closest_center_image[1][0]
                aruco_id = int(closest_center_image[2])
                if aruco_corners.shape[1] < 3:
                    aruco_corners = [aruco_corners]
                img = arucoAugmentImage(aruco_corners, img, augDic[aruco_id])

        # ------ Face recognition and comparison ------ #
        if (detectFace(img)[0]) and (len(arucoFound[0]) == 0):
            patient_face = augDic[aruco_id][0]
            result_face_recognition = face_recognition_compare(img, patient_face)
            img, mse, ifx, ify = result_face_recognition[0], result_face_recognition[1], result_face_recognition[2], \
                                 result_face_recognition[3]
            if mse > 10 ** (-5):
                cv2.putText(img, 'Right person', (int(ifx), int(ify)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 2)

            else:
                cv2.putText(img, 'Wrong person', (int(ifx), int(ify)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 2)

        # show image
        cv2.imshow("Image display", img)

        key = cv2.waitKey(1)
        # close the window
        if key == ord('q'):
            print('You pressed the quit key')
            break


if __name__ == '__main__':
    main()
