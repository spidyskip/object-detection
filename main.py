
import argparse
from yolo_opencv import yolo
import os
import logging
import numpy as np
from datetime import datetime

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
ap.add_argument('-o', '--out', required=False, default='out',
                help='path to output image')
args = ap.parse_args()



if __name__ == "__main__":

    logging.info('Start Processing...')

    # Input is a image or directory of images
    if '.' not in args.input:
        args.input = args.input + '/'

        logging.info(
            f'- Porcessing all images in the directory -> {args.input}')
        
        yolo = yolo(args)
        # Detect all image files in the folder
        with Progress() as progress:
            list = list([i for i in os.listdir(args.input) if i.lower().endswith(
                '.jpg') or i.lower().endswith('.png')])
            task1 = progress.add_task("[blue]Processing...", total=len(list))

            for filename in list:
                yolo.input = args.input[:args.input.find('/')+1] + filename
                yolo.filename = filename

                results = yolo.detect()
                yolo.draw(results)
                yolo.save(f'{yolo.filename.split(".")[0] }-object-detection.jpg')
                progress.update(task1, advance=1)

            logging.info(
                f'- Completed!')
    
    # Input is a single image
    elif args.input.lower().endswith('.jpg') or args.input.lower().endswith('.png'):
        
        logging.info(
            f'- Input is a single image : {args.input}')
        
        yolo = yolo(args)
        results = yolo.detect()
        yolo.draw(results.get_NMS())
        
        yolo.save(f'{yolo.filename.split(".")[0] }-object-detection.jpg')

        logging.info(
            f'- Completed!')
    # Input is a video
    else:
        logging.info(
            f'- Input is a video : {args.input}')
        name_video = args.input.split('/')[-1].split('.')[0]
        os.makedirs(args.out + '/' + name_video, exist_ok=True)
        args.out = args.out + '/' + name_video 

        # Read the video
        vidcap = cv2.VideoCapture(args.input)

        fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        
        logging.info(f'- Video duration: {duration} seconds'
                     f' - {frame_count} frames')

        success, image = vidcap.read()
        count = 0
        success = True
        yolo = yolo(args)
        with Progress() as progress:
            bar = progress.add_task("[red]Processing...", total=frame_count)
            while success:
                success, image = vidcap.read()
                
                logging.info(
                    f'- Read a new frame {count}: {success}')
                results = yolo.detect(image)
                yolo.draw(results)
    
                cv2.imwrite(args.out + '/' + "frame%d.jpg" %
                            count, yolo.image)     # save frame as JPEG file
                count += 1
                progress.update(bar, advance=1)

            logging.info(
                f'- Completed!')
        
    logging.info('Finished!')