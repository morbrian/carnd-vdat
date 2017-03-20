import lessons_functions as lf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import os
import os.path as path
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


class VehicleDetectionPipeline:
    """
    Class helps maintain configuration parameters and state for the detection pipeline.
    Provides method calls for all steps of the pipeline, many of which are wrappers
    used to call more specific functions from `lessons_functions.py`.
    """

    cars = None  # list of cars sample data to train with
    notcars = None  # list of notcars sample data to train with
    output_folder = None  # output folder for any data generated
    svc = None  # classifier
    scaler = None  # used for scaling feature vectors
    raw_bboxes = None  # all matching bboxes at any scale, includes overlapping boxes
    frame_heatmap = None  # heatmap for single frame
    historic_heatmap = None  # heatmap for combined history of frames
    fused_bboxes = None  # collection of bboxes with duplicate overlapping matches reduced to single boxes
    draw_raw = False  # use True to draw overlapping bounding boxes during image processing
    draw_raw_color = (155, 0, 0)  # draw single frame raw bboxes
    draw_fused_color = (0, 0, 255)  # draw fused history of bboxes
    calculate_frame_heatmap = False  # use True to draw heatmap during image processing
    calculate_historic_heatmap = True  # True to calculate historic heatmap
    draw_fused = True  # use True to draw fused bounding boxes during image processing
    frame_counter = 0  # used during video processing to identify frame number
    save_frame_range = None  # optionally save frames identified in this list
    orient = 9  # number of orientation bins (9 was consistently the best choice)
    pix_per_cell = 8  # number of pixels in height/width to size each cell square
    cell_per_block = 2  # number of cells in a block
    cells_per_step = 1  # number of cells to move search window in each step
    hist_bins = 32  # number of histogram bins when extracting features
    spatial_size = (32, 32)  # size of spatial window feature
    color_space = 'LUV'  # color space to use for feature extraction (LUV was consistently the best choice)
    ystart = 390  # where search windows should start
    ystop = 670  # where search windows should stop
    window_scales = [0.8, 1.0, 1.2, 1.5]  # list of scales to use for window search
    bbox_history = []  # history of bounding boxes identifying probably object detections
    bbox_history_limit = 10  # number of frame results to remember in history
    heatmap_frame_threshold = 1  # heat detection threshold in frame when deciding to include a box
    heatmap_historic_threshold = 8  # heat detection threshold combined over the history for box inclusion decision
    classifier_file = None
    scaler_file = None
    save_preview = True  # save all annotated video frames while processing

    def __init__(self, cars, notcars, output_folder, save_frame_range=None):
        """
        Initialize class
        :param cars: list of cars samples
        :param notcars: list of notcars samples
        :param output_folder: output_folder to create on filesystem
        :param save_frame_range: frame identifiers to save during video processing
        """
        self.cars = cars
        self.notcars = notcars
        self.save_frame_range = save_frame_range
        self.output_folder = output_folder
        self.classifier_file = '/'.join([output_folder, 'classifier.pkl'])
        self.scaler_file = '/'.join([output_folder, 'scaler.pkl'])

    def prepare_classifier(self, sample_count=5, test=False):
        """
        Preprocess sample data and train classifier.
        :param sample_count: number of samples to save for documentation purposes
        :param test: True of some of the data should be used for testing classifier
        """
        import pickle
        if path.exists(self.classifier_file):
            with open(self.classifier_file, 'rb') as fid:
                self.svc = pickle.load(fid)
            with open(self.scaler_file, 'rb') as fid:
                self.scaler = pickle.load(fid)

            print("loaded classifer and scaler from: {} and {}".format(self.classifier_file, self.scaler_file))
            return

        t = time.time()
        car_features = lf.extract_features_list(self.cars, color_space=self.color_space, orient=self.orient,
                                                pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                hog_channel='ALL',
                                                tag="car", vis_count=sample_count, vis_folder=self.output_folder)
        notcar_features = lf.extract_features_list(self.notcars, color_space=self.color_space, orient=self.orient,
                                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                   hog_channel='ALL',
                                                   tag="notcar", vis_count=sample_count, vis_folder=self.output_folder)
        print("Feature Extraction Duration: {} sec".format(time.time() - t))

        # get the same number of notcar features as we have for car features, to balance data
        notcar_features = shuffle(notcar_features)[:len(car_features)]

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scalerj
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        if test is True:
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.05, random_state=rand_state)
        else:
            X_train = scaled_X
            y_train = y

        # Use a linear SVC
        svc = LinearSVC(C=4.0, max_iter=2000)
        # Check the training time for the SVC
        print("Start training on {} training samples".format(len(X_train)))
        t = time.time()
        svc.fit(X_train, y_train)
        print(round(time.time() - t, 2), 'Seconds to train SVC...')

        if test is True:
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        self.svc = svc
        self.scaler = X_scaler

        with open(self.classifier_file, 'wb') as f:
            pickle.dump(svc, f)

        with open(self.scaler_file, 'wb') as f:
            pickle.dump(X_scaler, f)

        print("saved classifer and scaler to: {} and {}".format(self.classifier_file, self.scaler_file))

    def identify_raw_bboxes(self, image, grid=False):
        """
        Identify bounding boxes for the current frame.
        Processing includes a frame specific heatmap and fusing boxes, and result is included in history.
        :param image: image to search
        :param grid: True to include all boxes regardless of classifier prediction (useful for documentation)
        """
        bboxes = []
        for scale in self.window_scales:
            bboxes.extend(lf.find_cars(image, self.svc, self.scaler,
                                       ystart=self.ystart, ystop=self.ystop, scale=scale,
                                       orient=self.orient, pix_per_cell=self.pix_per_cell, cells_per_step=self.cells_per_step,
                                       cell_per_block=self.cell_per_block, spatial_size=self.spatial_size,
                                       hist_bins=self.hist_bins, grid=grid))

        # temporary store in raw_bboxes to prepare for additional processing
        self.raw_bboxes = bboxes

        # generate single frame heatmap and fused boxes
        self.generate_frame_heatmap(image)
        frame_fused = lf.fuse_bboxes(self.frame_heatmap)

        # save final fused result for frame
        self.raw_bboxes = frame_fused

        # remember this frame's bboxes in history
        self.bbox_history = [self.raw_bboxes] + self.bbox_history
        if len(self.bbox_history) > self.bbox_history_limit:
            self.bbox_history = self.bbox_history[:-1]

    def generate_frame_heatmap(self, image):
        """
        Generate heatmap for a single frame using raw_bboxes
        :param image: image to process
        """
        self.frame_heatmap = lf.produce_heatmap(image,  self.raw_bboxes, threshold=self.heatmap_frame_threshold)

    def generate_historic_heatmap(self, image):
        """
        Generate heatmap from history of bounding boxes
        :param image: image to process
        """
        historic = []
        for bbox in self.bbox_history:
            historic.extend(bbox)

        historic.extend(self.raw_bboxes)

        self.historic_heatmap = lf.produce_heatmap(image,  historic, threshold=self.heatmap_historic_threshold)

    def identify_fused_bboxes(self, heatmap=None):
        """
        Fuse the bounding boxes identified in a heatmap.
        Use heatmap if specified, otherwise if history buffer is nearly full use that, otherwise use frame_heatmap
        :param heatmap: heatmap to use
        """
        if heatmap is not None:
            self.fused_bboxes = lf.fuse_bboxes(heatmap)
        elif len(self.bbox_history) >= self.bbox_history_limit - 2:
            self.fused_bboxes = lf.fuse_bboxes(self.historic_heatmap)
        else:
            self.fused_bboxes = lf.fuse_bboxes(self.frame_heatmap)

    def apply_pipeline(self, image):
        """
        Apply the entire pipeline to a single image and return an annotated image.
        :param image: frame to process
        :return: annotated frame
        """
        self.frame_counter += 1

        # useful debugging condition, a range of troublesome images can optionally be saved to disk
        if self.save_frame_range is not None and self.frame_counter in self.save_frame_range:
            frame_output = '/'.join([self.output_folder, "frame{:05d}.jpg".format(self.frame_counter)])
            cv2.imwrite(frame_output, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("saved frame: {}".format(frame_output))

        self.identify_raw_bboxes(image)

        if self.calculate_historic_heatmap is True:
            self.generate_historic_heatmap(image)

        self.identify_fused_bboxes()

        if self.draw_raw is True and self.buffering() is False:
            image = lf.draw_bboxes(image, self.raw_bboxes, color=self.draw_raw_color, thick=5)

        if self.draw_fused is True and self.buffering() is False:
            image = lf.draw_bboxes(image, self.fused_bboxes, color=self.draw_fused_color, thick=8)

        if self.frame_counter > 0:
            cv2.putText(image, 'Frame({:05d})'.format(self.frame_counter), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if self.buffering() is True:
            cv2.putText(image, 'buffering...', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if self.save_preview is True:
            frame_output = '/'.join([self.output_folder, "preview{:05d}.jpg".format(self.frame_counter)])
            cv2.imwrite(frame_output, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return image

    def buffering(self):
        return len(self.bbox_history) < self.bbox_history_limit - 1

def load_training_data(path_patterns):
    """
    Organize all file names into cars or notcars.
    :param path_patterns:
    :return:
    """
    import glob
    cars = []
    notcars = []
    for pp in path_patterns:
        images = glob.glob(pp)
        for image in images:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)

    return cars, notcars


def demo_pipeline(pipeline, sample_images):
    """
    Demo the pipeline on the list of sample_images, storing various annotated images
    to help describe each step of the process.
    :param pipeline: pipeline object
    :param sample_images: list of sample images to process
    """
    pipeline.prepare_classifier(test=True)
    # read single frame sample to demo processes on
    for sample_image in sample_images:
        short_name = path.split(sample_image)[-1]
        if path.splitext(sample_image)[1] == ".png":
            image = cv2.imread(sample_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = mpimg.imread(sample_image)

        # show the search grid, setting grid=True returns every bbox regardless of matching
        # pipeline.identify_raw_bboxes(image, grid=True)
        # pipeline.bbox_history = []
        # grid_image = lf.draw_bboxes(np.copy(image), pipeline.raw_bboxes)
        # grid_filename = '/'.join([pipeline.output_folder, "search_grid_{}".format(short_name)])
        # mpimg.imsave(grid_filename, grid_image)
        # print("saved to: {}".format(grid_filename))

        # get overlapping bboxes
        pipeline.identify_raw_bboxes(image)
        raw_bbox_image = lf.draw_bboxes(np.copy(image), pipeline.raw_bboxes)

        # pipeline.generate_frame_heatmap(image)
        pipeline.generate_historic_heatmap(image)
        pipeline.identify_fused_bboxes()
        fused_image = lf.draw_bboxes(np.copy(image), pipeline.fused_bboxes, thick=6)

        fig = plt.figure()
        subplot = plt.subplot(141)
        subplot.axis('off')
        plt.imshow(raw_bbox_image)
        plt.title('Matches')
        subplot = plt.subplot(142)
        subplot.axis('off')
        plt.imshow(pipeline.frame_heatmap, cmap='hot')
        plt.title('Frame Heat')
        subplot = plt.subplot(143)
        subplot.axis('off')
        plt.imshow(pipeline.historic_heatmap, cmap='hot')
        plt.title('Historic Heat')
        subplot = plt.subplot(144)
        subplot.axis('off')
        plt.imshow(fused_image)
        plt.title('Fused')
        fig.tight_layout()
        search_sequence_filename = '/'.join([pipeline.output_folder, "search_sequence_{}".format(short_name)])
        plt.savefig(search_sequence_filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print("saved to: {}".format(search_sequence_filename))


def process_video(video_file, pipeline):
    """
    Process a video file and produce an annotated version of the video
    :param video_file: video to process
    :param pipeline: configured pipeline object to use for processing
    """
    if not path.exists(pipeline.output_folder):
        os.makedirs(pipeline.output_folder)
    pipeline.prepare_classifier()
    base_name = path.split(video_file)[1]
    vdat_output = '/'.join([pipeline.output_folder, "vdat_{}".format(base_name)])
    vclip = VideoFileClip(video_file, audio=False)
    vdat_clip = vclip.fl_image(pipeline.apply_pipeline)
    vdat_clip.write_videofile(vdat_output, audio=False)

    print("Processed video saved: {}".format(vdat_output))


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-a', '--activity', dest='activity', default='demo',
                      help="activity to perform [demo, video, all], to create demo images or process video or both")
    parser.add_option('-v', '--video_input', dest='video_input', default='./project_video.mp4',
                      help="video file to process.")
    parser.add_option('-t', '--training_patterns', dest='training_patterns',
                      default=['./training/large/*.png', './training/small/*.jpeg'],
                      help="path to folder of car and non-car images to use.")
    parser.add_option('-o', '--output_folder', dest='output_folder', default='./output_folder',
                      help="output folder to hold examples of images during process.")
    parser.add_option('--save_frame_range', dest='save_frame_range',
                      default=None,
                      help="min,max inclusive frame ids to save frames to disk.")

    options, args = parser.parse_args()
    activity = options.activity

    cars, not_cars = load_training_data(options.training_patterns)
    pipeline = VehicleDetectionPipeline(cars, not_cars,
                                        output_folder=options.output_folder,
                                        save_frame_range=options.save_frame_range)

    if 'all' == activity or 'demo' == activity:
        print("Demo pipeline components on sample images")
        demo_pipeline(pipeline, ["./samples/bbox-example-image.jpg"])

    if 'all' == activity or 'video' == activity:
        print("Process video file {}".format(options.video_input))
        process_video(options.video_input, pipeline)
        print("Video processing complete")


if __name__ == "__main__":
    main()
