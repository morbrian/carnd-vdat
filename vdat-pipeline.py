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
    draw_raw = True  # use True to draw overlapping bounding boxes during image processing
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
    window_scales = [0.8, 1.0, 1.2, 1.8]  # list of scales to use for window search
    bbox_history = []  # history of bounding boxes identifying probably object detections
    bbox_history_limit = 10  # number of frame results to remember in history
    heatmap_frame_threshold = 1  # heat detection threshold in frame when deciding to include a box
    heatmap_historic_threshold = 7  # heat detection threshold combined over the history for box inclusion decision
    classifier_file = None
    scaler_file = None

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

        # remember this frame's bboxes in history
        self.bbox_history = [self.fused_bboxes] + self.bbox_history
        if len(self.bbox_history) > self.bbox_history_limit:
            self.bbox_history = self.bbox_history[:-1]

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

        if self.draw_raw is True and self.buffering is False:
            image = lf.draw_bboxes(image, self.raw_bboxes, color=self.draw_raw_color, thick=5)

        if self.draw_fused is True and self.buffering is False:
            image = lf.draw_bboxes(image, self.fused_bboxes, color=self.draw_fused_color, thick=8)

        if self.frame_counter > 0:
            cv2.putText(image, 'Frame({:05d})'.format(self.frame_counter), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if self.buffering is True:
            cv2.putText(image, 'buffering...', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def buffering(self):
        return len(self.bbox_history) > self.bbox_history_limit - 2

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
    parser.add_option('-a', '--activity', dest='activity', default='video',
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
                          # "./samples/frame00001.jpg",
                          # "./samples/frame00002.jpg",
                          # "./samples/frame00003.jpg",
                          # "./samples/frame00004.jpg",
                          # "./samples/frame00005.jpg",
                          # "./samples/frame00006.jpg",
                          # "./samples/frame00007.jpg",
                          # "./samples/frame00008.jpg",
                          # "./samples/frame00009.jpg",
                          # "./samples/frame00010.jpg",
                          # "./samples/frame00011.jpg",
                          # "./samples/frame00012.jpg",
                          # "./samples/frame00013.jpg",
                          # "./samples/frame00014.jpg",
                          # "./samples/frame00015.jpg",
                          # "./samples/frame00016.jpg",
                          # "./samples/frame00017.jpg",
                          # "./samples/frame00018.jpg",
                          # "./samples/frame00019.jpg",
                          # "./samples/frame00020.jpg",
                          # "./samples/frame00021.jpg",
                          # "./samples/frame00022.jpg",
                          # "./samples/frame00023.jpg",
                          # "./samples/frame00024.jpg",
                          # "./samples/frame00025.jpg",
                          # "./samples/frame00026.jpg",
                          # "./samples/frame00027.jpg",
                          # "./samples/frame00028.jpg",
                          # "./samples/frame00029.jpg",
                          # "./samples/frame00030.jpg",
                          # "./samples/frame00031.jpg",
                          # "./samples/frame00032.jpg",
                          # "./samples/frame00033.jpg",
                          # "./samples/frame00034.jpg",
                          # "./samples/frame00035.jpg",
                          # "./samples/frame00036.jpg",
                          # "./samples/frame00037.jpg",
                          # "./samples/frame00038.jpg",
                          # "./samples/frame00039.jpg",
                          # "./samples/frame00040.jpg",
                          # "./samples/frame00041.jpg",
                          # "./samples/frame00042.jpg",
                          # "./samples/frame00043.jpg",
                          # "./samples/frame00044.jpg",
                          # "./samples/frame00045.jpg",
                          # "./samples/frame00046.jpg",
                          # "./samples/frame00047.jpg",
                          # "./samples/frame00048.jpg",
                          # "./samples/frame00049.jpg",
                          # "./samples/frame00050.jpg",
                          # "./samples/frame00051.jpg",
                          # "./samples/frame00052.jpg",
                          # "./samples/frame00053.jpg",
                          # "./samples/frame00054.jpg",
                          # "./samples/frame00055.jpg",
                          # "./samples/frame00056.jpg",
                          # "./samples/frame00057.jpg",
                          # "./samples/frame00058.jpg",
                          # "./samples/frame00059.jpg",
                          # "./samples/frame00060.jpg",
                          # "./samples/frame00061.jpg",
                          # "./samples/frame00062.jpg",
                          # "./samples/frame00063.jpg",
                          # "./samples/frame00064.jpg",
                          # "./samples/frame00065.jpg",
                          # "./samples/frame00066.jpg",
                          # "./samples/frame00067.jpg",
                          # "./samples/frame00068.jpg",
                          # "./samples/frame00069.jpg",
                          # "./samples/frame00070.jpg",
                          # "./samples/frame00071.jpg",
                          # "./samples/frame00072.jpg",
                          # "./samples/frame00073.jpg",
                          # "./samples/frame00074.jpg",
                          # "./samples/frame00075.jpg",
                          # "./samples/frame00076.jpg",
                          # "./samples/frame00077.jpg",
                          # "./samples/frame00078.jpg",
                          # "./samples/frame00079.jpg",
                          # "./samples/frame00080.jpg",
                          # "./samples/frame00081.jpg",
                          # "./samples/frame00082.jpg",
                          # "./samples/frame00083.jpg",
                          # "./samples/frame00084.jpg",
                          # "./samples/frame00085.jpg",
                          # "./samples/frame00086.jpg",
                          # "./samples/frame00087.jpg",
                          # "./samples/frame00088.jpg",
                          # "./samples/frame00089.jpg",
                          # "./samples/frame00090.jpg",
                          # "./samples/frame00091.jpg",
                          # "./samples/frame00092.jpg",
                          # "./samples/frame00093.jpg",
                          # "./samples/frame00094.jpg",
                          # "./samples/frame00095.jpg",
                          # "./samples/frame00096.jpg",
                          # "./samples/frame00097.jpg",
                          # "./samples/frame00098.jpg",
                          # "./samples/frame00099.jpg"
                          # "./samples/frame00400.jpg",
                          # "./samples/frame00401.jpg",
                          # "./samples/frame00402.jpg",
                          # "./samples/frame00403.jpg",
                          # "./samples/frame00404.jpg",
                          # "./samples/frame00405.jpg",
                          # "./samples/frame00406.jpg",
                          # "./samples/frame00407.jpg",
                          # "./samples/frame00408.jpg",
                          # "./samples/frame00409.jpg",
                          # "./samples/frame00410.jpg",
                          # "./samples/frame00411.jpg",
                          # "./samples/frame00412.jpg",
                          # "./samples/frame00413.jpg",
                          # "./samples/frame00414.jpg",
                          # "./samples/frame00415.jpg",
                          # "./samples/frame00416.jpg",
                          # "./samples/frame00417.jpg",
                          # "./samples/frame00418.jpg",
                          # "./samples/frame00419.jpg",
                          # "./samples/frame00420.jpg",
                          # "./samples/frame00421.jpg",
                          # "./samples/frame00422.jpg",
                          # "./samples/frame00423.jpg",
                          # "./samples/frame00424.jpg",
                          # "./samples/frame00425.jpg",
                          # "./samples/frame00426.jpg",
                          # "./samples/frame00427.jpg",
                          # "./samples/frame00428.jpg",
                          # "./samples/frame00429.jpg",
                          # "./samples/frame00430.jpg",
                          # "./samples/frame00431.jpg",
                          # "./samples/frame00432.jpg",
                          # "./samples/frame00433.jpg",
                          # "./samples/frame00434.jpg",
                          # "./samples/frame00435.jpg",
                          # "./samples/frame00436.jpg",
                          # "./samples/frame00437.jpg",
                          # "./samples/frame00438.jpg",
                          # "./samples/frame00439.jpg",
                          # "./samples/frame00440.jpg",
                          # "./samples/frame00441.jpg",
                          # "./samples/frame00442.jpg",
                          # "./samples/frame00443.jpg",
                          # "./samples/frame00444.jpg",
                          "./samples/frame00445.jpg",
                          "./samples/frame00446.jpg",
                          "./samples/frame00447.jpg",
                          "./samples/frame00448.jpg",
                          "./samples/frame00449.jpg",
                          "./samples/frame00450.jpg",
                          "./samples/frame00451.jpg",
                          "./samples/frame00452.jpg",
                          "./samples/frame00453.jpg",
                          "./samples/frame00454.jpg",
                          "./samples/frame00455.jpg",
                          "./samples/frame00456.jpg",
                          "./samples/frame00457.jpg",
                          "./samples/frame00458.jpg",
                          "./samples/frame00459.jpg",
                          "./samples/frame00460.jpg",
                          "./samples/frame00461.jpg",
                          "./samples/frame00462.jpg",
                          "./samples/frame00463.jpg",
                          "./samples/frame00464.jpg",
                          "./samples/frame00465.jpg",
                          "./samples/frame00466.jpg",
                          "./samples/frame00467.jpg",
                          "./samples/frame00468.jpg",
                          "./samples/frame00469.jpg",
                          "./samples/frame00470.jpg",
                          "./samples/frame00471.jpg",
                          "./samples/frame00472.jpg",
                          "./samples/frame00473.jpg",
                          "./samples/frame00474.jpg",
                          "./samples/frame00475.jpg",
                          "./samples/frame00476.jpg",
                          "./samples/frame00477.jpg",
                          "./samples/frame00478.jpg",
                          "./samples/frame00479.jpg",
                          "./samples/frame00480.jpg",
                          "./samples/frame00481.jpg",
                          "./samples/frame00482.jpg",
                          "./samples/frame00483.jpg",
                          "./samples/frame00484.jpg",
                          "./samples/frame00485.jpg",
                          "./samples/frame00486.jpg",
                          "./samples/frame00487.jpg",
                          "./samples/frame00488.jpg",
                          "./samples/frame00489.jpg",
                          "./samples/frame00490.jpg",
                          "./samples/frame00491.jpg",
                          "./samples/frame00492.jpg",
                          "./samples/frame00493.jpg",
                          "./samples/frame00494.jpg",
                          "./samples/frame00495.jpg",
                          "./samples/frame00496.jpg",
                          "./samples/frame00497.jpg",
                          "./samples/frame00498.jpg",
                          "./samples/frame00499.jpg",
                          "./samples/frame00500.jpg",
                          "./samples/frame00501.jpg",
                          "./samples/frame00502.jpg",
                          "./samples/frame00503.jpg",
                          "./samples/frame00504.jpg",
                          "./samples/frame00505.jpg",
                          "./samples/frame00506.jpg",
                          "./samples/frame00507.jpg",
                          "./samples/frame00508.jpg",
                          "./samples/frame00509.jpg",
                          "./samples/frame00510.jpg",
                          "./samples/frame00511.jpg",
                          "./samples/frame00512.jpg",
                          "./samples/frame00513.jpg",
                          "./samples/frame00514.jpg",
                          "./samples/frame00515.jpg",
                          "./samples/frame00516.jpg",
                          "./samples/frame00517.jpg",
                          "./samples/frame00518.jpg",
                          "./samples/frame00519.jpg",
                          "./samples/frame00520.jpg",
                          "./samples/frame00521.jpg",
                          "./samples/frame00522.jpg",
                          "./samples/frame00523.jpg",
                          "./samples/frame00524.jpg",
                          "./samples/frame00525.jpg",
                          "./samples/frame00526.jpg",
                          "./samples/frame00527.jpg",
                          "./samples/frame00528.jpg",
                          "./samples/frame00529.jpg",
                          "./samples/frame00530.jpg",
                          "./samples/frame00531.jpg",
                          "./samples/frame00532.jpg",
                          "./samples/frame00533.jpg",
                          "./samples/frame00534.jpg",
                          "./samples/frame00535.jpg",
                          "./samples/frame00536.jpg",
                          "./samples/frame00537.jpg",
                          "./samples/frame00538.jpg",
                          "./samples/frame00539.jpg",
                          "./samples/frame00540.jpg",
                          "./samples/frame00541.jpg",
                          "./samples/frame00542.jpg",
                          "./samples/frame00543.jpg",
                          "./samples/frame00544.jpg",
                          "./samples/frame00545.jpg",
                          "./samples/frame00546.jpg",
                          "./samples/frame00547.jpg",
                          "./samples/frame00548.jpg",
                          "./samples/frame00549.jpg",
                          "./samples/frame00550.jpg",
                          "./samples/frame00551.jpg",
                          "./samples/frame00552.jpg",
                          "./samples/frame00553.jpg",
                          "./samples/frame00554.jpg",
                          "./samples/frame00555.jpg",
                          "./samples/frame00556.jpg",
                          "./samples/frame00557.jpg",
                          "./samples/frame00558.jpg",
                          "./samples/frame00559.jpg",
                          "./samples/frame00560.jpg",
                          "./samples/frame00561.jpg",
                          "./samples/frame00562.jpg",
                          "./samples/frame00563.jpg",
                          "./samples/frame00564.jpg",
                          "./samples/frame00565.jpg",
                          "./samples/frame00566.jpg",
                          "./samples/frame00567.jpg",
                          "./samples/frame00568.jpg",
                          "./samples/frame00569.jpg",
                          "./samples/frame00570.jpg",
                          "./samples/frame00571.jpg",
                          "./samples/frame00572.jpg",
                          "./samples/frame00573.jpg",
                          "./samples/frame00574.jpg",
                          "./samples/frame00575.jpg",
                          "./samples/frame00576.jpg",
                          "./samples/frame00577.jpg",
                          "./samples/frame00578.jpg",
                          "./samples/frame00579.jpg",
                          "./samples/frame00580.jpg",
                          "./samples/frame00581.jpg",
                          "./samples/frame00582.jpg",
                          "./samples/frame00583.jpg",
                          "./samples/frame00584.jpg",
                          "./samples/frame00585.jpg",
                          "./samples/frame00586.jpg",
                          "./samples/frame00587.jpg",
                          "./samples/frame00588.jpg",
                          "./samples/frame00589.jpg",
                          "./samples/frame00590.jpg",
                          "./samples/frame00591.jpg",
                          "./samples/frame00592.jpg",
                          "./samples/frame00593.jpg",
                          "./samples/frame00594.jpg",
                          "./samples/frame00595.jpg",
                          "./samples/frame00596.jpg",
                          "./samples/frame00597.jpg",
                          "./samples/frame00598.jpg",
                          "./samples/frame00599.jpg",
                          "./samples/frame00600.jpg",
                          "./samples/frame00601.jpg",
                          "./samples/frame00602.jpg",
                          "./samples/frame00603.jpg",
                          "./samples/frame00604.jpg",
                          "./samples/frame00605.jpg",
                          "./samples/frame00606.jpg",
                          "./samples/frame00607.jpg",
                          "./samples/frame00608.jpg",
                          "./samples/frame00609.jpg",
                          "./samples/frame00610.jpg",
                          "./samples/frame00611.jpg",
                          "./samples/frame00612.jpg",
                          "./samples/frame00613.jpg",
                          "./samples/frame00614.jpg",
                          "./samples/frame00615.jpg",
                          "./samples/frame00616.jpg",
                          "./samples/frame00617.jpg",
                          "./samples/frame00618.jpg",
                          "./samples/frame00619.jpg",
                          "./samples/frame00620.jpg",
                          "./samples/frame00621.jpg",
                          "./samples/frame00622.jpg",
                          "./samples/frame00623.jpg",
                          "./samples/frame00624.jpg",
                          "./samples/frame00625.jpg",
                          "./samples/frame00626.jpg",
                          "./samples/frame00627.jpg",
                          "./samples/frame00628.jpg",
                          "./samples/frame00629.jpg",
                          "./samples/frame00630.jpg",
                          "./samples/frame00631.jpg",
                          "./samples/frame00632.jpg",
                          "./samples/frame00633.jpg",
                          "./samples/frame00634.jpg",
                          "./samples/frame00635.jpg",
                          "./samples/frame00636.jpg",
                          "./samples/frame00637.jpg",
                          "./samples/frame00638.jpg",
                          "./samples/frame00639.jpg",
                          "./samples/frame00640.jpg",
                          "./samples/frame00641.jpg",
                          "./samples/frame00642.jpg",
                          "./samples/frame00643.jpg",
                          "./samples/frame00644.jpg",
                          "./samples/frame00645.jpg",
                          "./samples/frame00646.jpg",
                          "./samples/frame00647.jpg",
                          "./samples/frame00648.jpg",
                          "./samples/frame00649.jpg",
                          "./samples/frame00650.jpg",
                          "./samples/frame00651.jpg",
                          "./samples/frame00652.jpg",
                          "./samples/frame00653.jpg",
                          "./samples/frame00654.jpg",
                          "./samples/frame00655.jpg",
                          "./samples/frame00656.jpg",
                          "./samples/frame00657.jpg",
                          "./samples/frame00658.jpg",
                          "./samples/frame00659.jpg",
                          "./samples/frame00660.jpg",
                          "./samples/frame00661.jpg",
                          "./samples/frame00662.jpg",
                          "./samples/frame00663.jpg",
                          "./samples/frame00664.jpg",
                          "./samples/frame00665.jpg",
                          "./samples/frame00666.jpg",
                          "./samples/frame00667.jpg",
                          "./samples/frame00668.jpg",
                          "./samples/frame00669.jpg",
                          "./samples/frame00670.jpg",
                          "./samples/frame00671.jpg",
                          "./samples/frame00672.jpg",
                          "./samples/frame00673.jpg",
                          "./samples/frame00674.jpg",
                          "./samples/frame00675.jpg",
                          "./samples/frame00676.jpg",
                          "./samples/frame00677.jpg",
                          "./samples/frame00678.jpg",
                          "./samples/frame00679.jpg"
                      ])

    if 'all' == activity or 'video' == activity:
        print("Process video file {}".format(options.video_input))
        process_video(options.video_input, pipeline)
        print("Video processing complete")


if __name__ == "__main__":
    main()
