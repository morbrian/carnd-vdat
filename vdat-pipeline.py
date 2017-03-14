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
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

class VehicleDetectionPipeline:

    svc = None  # classifier
    scaler = None  # used for scaling feature vectors
    raw_bboxes = None  # all matching bboxes at any scale, includes overlapping boxes
    heatmap = None  # heatmap of weighted locations of overlapping matches
    fused_bboxes = None  # collection of bboxes with duplicate overlapping matches reduced to single boxes
    draw_raw = False  # use True to draw overlapping bounding boxes during image processing
    draw_heatmap = False  # use True to draw heatmap during image processing
    draw_fused = True  # use True to draw fused bounding boxes during image processing

    def __init__(self):
        pass

    def prepare_classifier(self, cars, notcars, color_space='LUV', orient=9, pix_per_cell=8,
                           cell_per_block=2, hog_channel='ALL', output_folder=None, sample_count=5):
        t = time.time()
        car_features = lf.extract_features_list(cars, color_space=color_space, orient=orient,
                                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                hog_channel=hog_channel,
                                                tag="car", vis_count=sample_count, vis_folder=output_folder)
        notcar_features = lf.extract_features_list(notcars, color_space=color_space, orient=orient,
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

        self.svc = svc
        self.scaler = X_scaler

    def identify_raw_bboxes(self, image, grid=False):
        self.raw_bboxes = lf.find_cars(image, self.svc, self.scaler, grid=grid)

    def generate_heatmap(self, image):
        self.heatmap = lf.produce_heatmap(image,  self.raw_bboxes)

    def identify_fused_bboxes(self):
        self.fused_bboxes = lf.fuse_bboxes(self.heatmap)

    def apply_pipeline(self, image):
        self.identify_raw_bboxes(image)
        self.generate_heatmap(image)
        self.identify_fused_bboxes()

        if self.draw_heatmap is True:
            pass

        if self.draw_raw is True:
            image = lf.draw_bboxes(image, self.raw_bboxes, thick=6)

        if self.draw_fused is True:
            image = lf.draw_bboxes(image, self.fused_bboxes, thick=6)

        return image


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


def demo_pipeline(training_pattern, output_folder, sample_image):
    cars, not_cars = load_training_data(training_pattern)
    print("car sample count: {}".format(len(cars)))
    print("not_car sample count: {}".format(len(not_cars)))

    pipeline = VehicleDetectionPipeline()
    # train classifier on training data, also saves a few samples to the output folder
    pipeline.prepare_classifier(cars, not_cars, output_folder=output_folder)

    # read single frame sample to demo processes on
    image = mpimg.imread(sample_image)

    # show the search grid, setting grid=True returns every bbox regardless of matching
    pipeline.identify_raw_bboxes(image, grid=True)
    grid_image = lf.draw_bboxes(np.copy(image), pipeline.raw_bboxes)
    grid_filename = '/'.join([output_folder, "search_grid.jpg"])
    mpimg.imsave(grid_filename, grid_image)
    print("saved to: {}".format(grid_filename))

    # get overlapping bboxes
    pipeline.identify_raw_bboxes(image)
    raw_bbox_image = lf.draw_bboxes(np.copy(image), pipeline.raw_bboxes)

    pipeline.generate_heatmap(image)
    pipeline.identify_fused_bboxes()
    fused_image = lf.draw_bboxes(np.copy(image), pipeline.fused_bboxes, thick=6)

    fig = plt.figure()
    subplot = plt.subplot(131)
    subplot.axis('off')
    plt.imshow(raw_bbox_image)
    plt.title('Matches')
    subplot = plt.subplot(132)
    subplot.axis('off')
    plt.imshow(pipeline.heatmap, cmap='hot')
    plt.title('Heat Map')
    subplot = plt.subplot(133)
    subplot.axis('off')
    plt.imshow(fused_image)
    plt.title('Fused')
    fig.tight_layout()
    search_sequence_filename = '/'.join([output_folder, "search_sequence.jpg"])
    plt.savefig(search_sequence_filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("saved to: {}".format(search_sequence_filename))


def process_video(video_file, pipeline, output_folder, save_frame_range):
    if not path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = path.split(video_file)[1]
    vdat_output = '/'.join([output_folder, "vdat_{}".format(base_name)])
    vclip = VideoFileClip(video_file, audio=False)
    vdat_clip = vclip.fl_image(pipeline.apply_pipeline)
    vdat_clip.write_videofile(vdat_output, audio=False)

    print("Processed video saved: {}".format(vdat_output))


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-a', '--activity', dest='activity', default='video',
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
    activity = options.activity

    if 'all' == activity or 'demo' == activity:
        print("Demo pipeline components on sample images")
        demo_pipeline(options.training_pattern, options.output_folder, "./samples/bbox-example-image.jpg")

    if 'all' == activity or 'video' == activity:
        video_file = options.video_input
        # train on sample data
        cars, not_cars = load_training_data(options.training_pattern)
        pipeline = VehicleDetectionPipeline()
        pipeline.prepare_classifier(cars, not_cars, output_folder=options.output_folder)
        print("Process video file {}".format(video_file))
        process_video(video_file, pipeline, options.output_folder, save_frame_range=options.save_frame_range)
        print("Video processing complete")


if __name__ == "__main__":
    main()
