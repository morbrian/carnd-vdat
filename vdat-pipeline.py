import lessons_functions as lf
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import time
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import os
import os.path as path


def load_training_data(path_pattern):
    import glob
    images = glob.glob(path_pattern)
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars, notcars


def prepare_classifier(cars, notcars, color_space='LUV', orient=9, pix_per_cell=8,
                       cell_per_block=2, hog_channel='ALL', output_folder=None, sample_count=5):

    t = time.time()
    car_features = lf.extract_features(cars, color_space=color_space, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       tag="car", vis_count=sample_count, vis_folder=output_folder)
    notcar_features = lf.extract_features(notcars, color_space=color_space, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          hog_channel=hog_channel,
                                          tag="notcar", vis_count=sample_count, vis_folder=output_folder)
    print("Feature Extraction Duration: {} sec".format(time.time() - t))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time() - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-a', '--activity', dest='activity', default='all',
                      help="activity to perform [demo, video, all], to create demo images or process video or both")
    parser.add_option('-v', '--video_input', dest='video_input', default='./project_video.mp4',
                      help="video file to process.")
    parser.add_option('-t', '--training_pattern', dest='training_pattern', default='./training/small/*.jpeg',
                      help="path to folder of car and non-car images to use.")
    parser.add_option('-o', '--output_folder', dest='output_folder', default='./output_folder',
                      help="output folder to hold examples of images during process.")
    parser.add_option('--save_frame_range', dest='save_frame_range', default=None,
                      help="min,max inclusive frame ids to save frames to disk.")

    options, args = parser.parse_args()

    cars, not_cars = load_training_data(options.training_pattern)

    print("cars: {}".format(len(cars)))
    print("not_cars: {}".format(len(not_cars)))

    prepare_classifier(cars, not_cars, output_folder=options.output_folder)


if __name__ == "__main__":
    main()