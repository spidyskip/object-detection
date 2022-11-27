
import argparse
from yolo_opencv import yolo
import os
import logging
import numpy as np
import skvideo.io

import cv2
from rich.progress import Progress

range = 100

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='path to input image')
ap.add_argument('-model', '--model', required=True,
                help='path to model - weights, cfg')
ap.add_argument('-c', '--config', required=False,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=False,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
ap.add_argument('--search', required=False, default=None,
                help=f'look for a specific class')
ap.add_argument('--create', required=False, default=False,
                help=f'boolean to create a video')
ap.add_argument('-o', '--out', required=False, default='out',
                help='path to output image')
args = ap.parse_args()

def folder():
    global yolo, list
    args.input = args.input + '/'

    logging.info(
        f' Processing all images in the directory {args.input}')

    yolo = yolo(args)
    # Detect all image files in the folder
    with Progress() as progress:
        list = list([i for i in os.listdir(args.input) if i.lower().endswith(
            '.jpg') or i.lower().endswith('.png')])
        task1 = progress.add_task("[blue]Processing...", total=len(list))

        for filename in list:
            yolo.input = args.input[:args.input.find('/')+1] + filename
            yolo.filename = filename

            image = cv2.imread(yolo.input)

            results = yolo.detect(image)
            image_detected = results.draw(image, yolo.classes)

            yolo.save(image_detected,
                        f'{yolo.filename.split(".")[0]}-object-detection.jpg')

            progress.update(task1, advance=1)

        logging.info(
            f' Completed!')

def image():
    global yolo
    logging.info(
        f' Input is a single image : {args.input}')

    yolo = yolo(args)
    results = yolo.detect()
    image = cv2.imread(yolo.input)

    results = yolo.detect(image)
    image_detected = results.draw(image, yolo.classes)
    yolo.save(image_detected,
              f'{yolo.filename.split(".")[0]}-object-detection.jpg')

    logging.info(
        f' Completed!')

def video():
    global yolo
    logging.info(
        f' Input is a video : {args.input}')
    name_video = args.input.split('/')[-1].split('.')[0]
    args.out = os.path.join(args.out, name_video)
    os.makedirs(args.out, exist_ok=True)

    # Read the video
    vidcap = cv2.VideoCapture(args.input)

    # OpenCV v2.x used "CV_CAP_PROP_FPS"
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    logging.info(f' Video duration: {duration} seconds'
                    f' - {frame_count} frames - {round(fps,2)} fps')

    success, image = vidcap.read()
    count = 0
    success = True
    yolo = yolo(args)
    if args.create:
        out_video = np.empty([frame_count,
                              464, 848, 3], dtype=np.uint8)
        out_video = out_video.astype(np.uint8)

    with Progress() as progress:
        bar = progress.add_task("[red]Processing...", total=frame_count)
        while success :
            success, image = vidcap.read()

            logging.info(
                f' Read a new frame {count}: {success}')
            try:
                results = yolo.detect(image)
            except AttributeError:
                break

            image_detected = results.draw(image, yolo.classes, yolo.colors)
            
            if args.create:
                try:
                    out_video[count] = image_detected
                except:
                    pass
            else:
                cv2.imwrite(os.path.join(args.out, "frame%d.jpg" %
                                     count), image_detected)     # save frame as JPEG file
            
            count += 1
            progress.update(bar, advance=1)

        if args.create:
            # Writes the the output image sequences in a video file
            skvideo.io.vwrite(os.path.join(yolo.out.split('/')[0], f"{yolo.filename.split('.')[0]}-detected.mp4"), out_video)

        logging.info(
            f' Completed!')

if __name__ == "__main__":
    
    logging.info('Start Processing...')
    os.makedirs(args.out, exist_ok=True) # Create output folder
    
    # Input is a image or directory of images
    if os.path.isdir(args.input):
        folder()  

    # Input is a single image
    elif args.input.lower().endswith('.jpg') or args.input.lower().endswith('.png'):
        image()
        
    # Input is a video
    else:
        video()