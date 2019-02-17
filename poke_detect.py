##############################################################################
# poke_detect.py - Computer vision program to detect shiny pokemon in the 
#                  Nintendo Switch Pokemon: Let's Go games. Once the shiny
#                  pokemon is detected a sound will output from the computer
#                  to alert the user to go catch that pokemon! :D
#
# Copyright (C) 2019   Humza Syed
#
# This is free software so please use it however you'd like! :)
##############################################################################

import cv2
import numpy as np
import argparse
import os


"""
##############################################################################
# YOLOv3 Shiny Pokemon detection
##############################################################################
"""
def yolo_poke_detection(imgsPath, yolo_path, weights_file, classes_file):
    """
    Detects shiny pokemon in videos and draws a bounding box around the pokemon

    :param imgsPath:    Path to the input images directory
    :param yolo_path:   Path to the yolo algorithm's config file
    :param weights_file:Pre-trained face weights for yolo algorithm 
    :param classes_file:Classes text file for yolo algorithm 

    :returns:

    (Doesn't return anything except outputted images to the directory 'output_images')
    """
    # keep track of how many images processed on
    counter = 0

    for image_file in (os.listdir(imgsPath)):
        
        # image path
        image_path = os.path.join(imgsPath, image_file)

        # read input image
        image = cv2.imread(image_path)

        # get width and height
        height = image.shape[0]
        width  = image.shape[1]
        scale = 0.00392

        # get class names
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different color bounding boxes for different classes
        #colors_list = np.random.uniform(0, 255, size=(len(classes), 3))

        # temporarily commenting out colors_list above since we just have 1 class which is a face
        colors_list = [(0, 255, 0)]

        # read pre-trained model and config file to create network
        net = cv2.dnn.readNet(weights_file, yolo_path)

        # Prepare image to run through network
        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # run inference
        outs = net.forward(get_output_layers(net))

        # initializations for detection
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.35 #0.5
        nms_threshold = 0.4

        # get confidences, bounding box params, class_ids for each detection
        # ignore weak detections (under 0.5 confidence)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if(confidence > 0.35):#0.5):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w/2
                    y = center_y - h/2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppresion
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # detections after nms
        for ind in indices:
            ind = ind[0]
            box = boxes[ind]
            x = max(0, round(box[0]))
            y = max(0, round(box[1]))
            w = max(0, round(box[2]))
            h = max(0, round(box[3]))

            # draw the bounding box
            draw_bounding_box(image, classes, class_ids[ind], colors_list, x, y, w, h)

        # strip file extension of original image so we can write similar output image
        # i.e.) people.png --> people_output.png
        image_file = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(os.getcwd(), 'out_images_yolo', (image_file + '_output.png')), image)
        cv2.destroyAllWindows()
    
        print('Image {} was completed!'.format(counter))
        counter += 1

    print('Processed all {} images! :D'.format(counter))


def get_output_layers(net):
    """
    Obtain output layer names in the architecture

    :param   net: yolo network

    :return  output_layers: output layers used in the net
    """

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bounding_box(image, classes, class_id, colors_list, x, y, w, h):
    """
    Draw bounding boxes in image based off object detected and apply a blur to the inside of the box

    :param  image:      read in image using opencv
    :param  classes:    list of classes
    :param  class_id:   specific class id
    :param  colors:     colors used for bounding box
    :param  x, y, w, h: dimensions of image

    :return (the class label and colored bounding box on the image)

    """

    # get label and color for class
    label = str(classes[class_id])
    color = colors_list[class_id]
    
    # draw bounding box with text over it
    cv2.rectangle(image, (x,y), ((x+w), (y+h)), color, 2)
    cv2.putText(image, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


"""
##############################################################################
# Parser
##############################################################################
"""
def create_parser():
    """
    Function to take in input arguments from the user

    return: parser inputs
    """
    parser = argparse.ArgumentParser(
        description='Pokemon_Lets_Go_Shiny_Hunting arguments.')

    # function to allow parsing of true/false boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    #parser.add_argument('--img_path', type=str, default='test.jpeg',
            #help='Uses path of image; default=test.jpeg')
    parser.add_argument('-i', '--img_dir_path', type=str, default='images',
            help='Uses the provided directory name as the target directory where all input images are; default=images')
    
    # arguments for yolo face detection parameters
    parser.add_argument('-y', '--yolo_path', type=str, default='yolov3-face.cfg',
            help='Uses the provided path to the yolo config file; default=yolov3-face.cfg')
    parser.add_argument('-w', '--weights', type=str, default='yolov3-wider_16000.weights',
            help='Uses the provided path to the weights file for yolo; default=yolov3-wider_16000.weights')
    parser.add_argument('-c', '--classes', type=str, default='yolov3_classes.txt',
            help='Uses the provided path to the classes text file for yolo; default=yolov3_classes.txt')

    args = parser.parse_args()

    return args


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
def main(): 
    """
    Takes in input image to detect faces and then apply a blur like effect of some kind
    """

    # load parsed arguments
    args = create_parser()

    # join input image path
    imgsPath = os.path.join(os.getcwd(), args.img_dir_path)
    #print(imgsPath)

    # check which detection method we're using and creat folder for it if it doesn't exist already
    # then run face detection
    # Haar cascade detection
    # YOLO detection
    yolo_face_detection(imgsPath, args.yolo_path, args.weights, args.classes)

if __name__== '__main__':
    main()
