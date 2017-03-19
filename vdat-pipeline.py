import lessons_functions as lf
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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

    cars = None
    notcars = None
    output_folder = None  # output folder for any data generated
    svc = None  # classifier
    scaler = None  # used for scaling feature vectors
    raw_bboxes = None  # all matching bboxes at any scale, includes overlapping boxes
    frame_heatmap = None  # heatmap of weighted locations of overlapping matches
    historic_heatmap = None  #
    fused_bboxes = None  # collection of bboxes with duplicate overlapping matches reduced to single boxes
    draw_raw = True  # use True to draw overlapping bounding boxes during image processing
    draw_raw_color = (255, 0, 0)
    draw_fused_color = (0, 0, 255)
    calculate_frame_heatmap = False  # use True to draw heatmap during image processing
    calculate_historic_heatmap = True
    draw_fused = True  # use True to draw fused bounding boxes during image processing
    frame_counter = 0
    save_frame_range = None
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    cells_per_step = 1
    hist_bins = 32
    spatial_size = (32, 32)
    color_space = 'LUV'
    ystart = 390
    ystop = 670
    window_scales = [0.8, 1.2, 1.8]
    bbox_history = []
    bbox_history_limit = 30
    heatmap_frame_threshold = 1
    heatmap_historic_threshold = 20

    def __init__(self, cars, notcars, output_folder, save_frame_range=None):
        self.cars = cars
        self.notcars = notcars
        self.save_frame_range = save_frame_range
        self.output_folder = output_folder

    def prepare_classifier(self, sample_count=5, test=False):
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
                scaled_X, y, test_size=0.2, random_state=rand_state)
            # pool_size = len(y)
            # test_size = int(pool_size / 5)
            # X_train, y_train = shuffle(scaled_X[test_size:], y[test_size:])
            # X_test, y_test = scaled_X[:test_size], y[:test_size]
            # print("train size: {}, {}".format(len(X_train), len(y_train)))
            # print("test size: {}, {}".format(len(X_test), len(y_test)))
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

    def identify_raw_bboxes(self, image, grid=False):
        bboxes = []
        for scale in self.window_scales:
            bboxes.extend(lf.find_cars(image, self.svc, self.scaler,
                                       ystart=self.ystart, ystop=self.ystop, scale=scale,
                                       orient=self.orient, pix_per_cell=self.pix_per_cell, cells_per_step=self.cells_per_step,
                                       cell_per_block=self.cell_per_block, spatial_size=self.spatial_size,
                                       hist_bins=self.hist_bins, grid=grid))

        self.bbox_history = [bboxes] + self.bbox_history
        if len(self.bbox_history) > self.bbox_history_limit:
            self.bbox_history = self.bbox_history[:-1]
        self.raw_bboxes = bboxes

    def generate_frame_heatmap(self, image):
        self.frame_heatmap = lf.produce_heatmap(image,  self.raw_bboxes, threshold=self.heatmap_frame_threshold)

    def generate_historic_heatmap(self, image):
        historic = []
        for bbox in self.bbox_history:
            historic.extend(bbox)
        self.historic_heatmap = lf.produce_heatmap(image,  historic, threshold=self.heatmap_historic_threshold)

    def identify_fused_bboxes(self):
        self.fused_bboxes = lf.fuse_bboxes(self.historic_heatmap)

    def apply_pipeline(self, image):
        self.frame_counter += 1

        # useful debugging condition, a range of troublesome images can optionally be saved to disk
        if self.save_frame_range is not None and self.frame_counter in self.save_frame_range:
            frame_output = '/'.join([self.output_folder, "frame{:05d}.jpg".format(self.frame_counter)])
            cv2.imwrite(frame_output, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("saved frame: {}".format(frame_output))

        self.identify_raw_bboxes(image)

        if self.calculate_frame_heatmap is True:
            self.generate_frame_heatmap(image)

        if self.calculate_historic_heatmap is True:
            self.generate_historic_heatmap(image)

        self.identify_fused_bboxes()

        if self.draw_raw is True:
            image = lf.draw_bboxes(image, self.raw_bboxes, color=self.draw_raw_color, thick=5)

        if self.draw_fused is True:
            image = lf.draw_bboxes(image, self.fused_bboxes, color=self.draw_fused_color, thick=8)

        if self.frame_counter > 0:
            cv2.putText(image, 'Frame({:05d})'.format(self.frame_counter), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image


def load_training_data(path_patterns):
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

        pipeline.generate_frame_heatmap(image)
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
        demo_pipeline(pipeline,
                      [
                       # "./samples/frame00020.jpg", "./samples/frame00021.jpg", "./samples/frame00022.jpg", "./samples/frame00023.jpg", "./samples/frame00024.jpg",
                       # "./samples/frame00025.jpg", "./samples/frame00026.jpg", "./samples/frame00027.jpg", "./samples/frame00028.jpg", "./samples/frame00029.jpg",
                       # "./samples/frame00030.jpg", "./samples/frame00031.jpg", "./samples/frame00032.jpg", "./samples/frame00033.jpg", "./samples/frame00034.jpg",
                       # "./samples/frame00035.jpg", "./samples/frame00036.jpg", "./samples/frame00037.jpg", "./samples/frame00038.jpg", "./samples/frame00039.jpg",
                       # "./samples/frame00040.jpg"
                       "./samples/frame00300.jpg", "./samples/frame00301.jpg", "./samples/frame00302.jpg", "./samples/frame00303.jpg", "./samples/frame00304.jpg",
                       "./samples/frame00305.jpg", "./samples/frame00306.jpg", "./samples/frame00307.jpg", "./samples/frame00308.jpg", "./samples/frame00309.jpg",
                       "./samples/frame00310.jpg", "./samples/frame00311.jpg", "./samples/frame00312.jpg", "./samples/frame00313.jpg", "./samples/frame00314.jpg",
                       "./samples/frame00315.jpg", "./samples/frame00316.jpg", "./samples/frame00317.jpg", "./samples/frame00318.jpg", "./samples/frame00319.jpg",
                       "./samples/frame00320.jpg"
                       # "./samples/frame00460.jpg", "./samples/frame00461.jpg", "./samples/frame00462.jpg", "./samples/frame00463.jpg", "./samples/frame00464.jpg",
                       # "./samples/frame00465.jpg", "./samples/frame00466.jpg", "./samples/frame00467.jpg", "./samples/frame00468.jpg", "./samples/frame00469.jpg",
                       # "./samples/frame00470.jpg", "./samples/frame00471.jpg", "./samples/frame00472.jpg", "./samples/frame00473.jpg", "./samples/frame00474.jpg",
                       # "./samples/frame00475.jpg", "./samples/frame00476.jpg", "./samples/frame00477.jpg", "./samples/frame00478.jpg", "./samples/frame00479.jpg",
                       # "./samples/frame00480.jpg"
                       ])

    if 'all' == activity or 'video' == activity:
        print("Process video file {}".format(options.video_input))
        process_video(options.video_input, pipeline)
        print("Video processing complete")


if __name__ == "__main__":
    main()
