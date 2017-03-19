import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import os
import os.path as path
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


def save_hog_sequence(output_image_name, title, image, hog_vis, file_features):
    """
    save the image to the output file
    :param output_image_name: file name of image describing hog processing
    :param title: title of image
    :param image: image data
    :param hog_vis: single or array of hog channels
    :param file_features: other features, like spatial or histo
    """
    hog_vis_count = 1 if len(hog_vis) > 3 else 3
    row_size = hog_vis_count + 2
    fig = plt.figure(figsize=(25, 4))

    subplot = plt.subplot(1, row_size, 1)
    subplot.axis('off')
    subplot.set_title(title)
    plt.imshow(image)

    if hog_vis_count == 1:
        subplot = plt.subplot(1, row_size, 2)
        subplot.axis('off')
        subplot.set_title('single channel hog')
        plt.imshow(hog_vis, cmap='gray')
    else:
        for i in range(3):
            subplot = plt.subplot(1, row_size, i + 2)
            subplot.axis('off')
            subplot.set_title('hog ch{}'.format(i))
            plt.imshow(hog_vis[i], cmap='gray')

    subplot = plt.subplot(1, row_size, row_size)
    subplot.set_ylim([0, 500])
    subplot.set_title('color+spatial feats')
    subplot.plot(file_features)

    plt.savefig(output_image_name, bbox_inches='tight', dpi=50)
    plt.close(fig)
    print("saved to: {}".format(output_image_name))


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Produce HOG features using skimage and optionally return visual representations.
    :param img: image to process
    :param orient: number of orientation summary cells
    :param pix_per_cell: number of pixels to include
    :param cell_per_block: number of cells in each block
    :param vis: use True to generate the hog visual representation
    :param feature_vec: True if the returned vector should be raveled
    :return:
    """
    if vis is True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def convert_color(image, conv):
    if conv == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif conv == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif conv == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image_file, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True,
                     tag='', vis=False, vis_folder=None):
    file_features = []
    vis_features = []
    hog_vis = None
    if vis is True and vis_folder is not None:
        if not path.exists(vis_folder):
            os.makedirs(vis_folder)

    bins_range = (0, 255)
    # bins_range = (0, 1.0)
    file_extension = path.splitext(image_file)[1]
    if file_extension == ".png":
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif file_extension == ".jpg" or file_extension == ".jpeg":
        image = mpimg.imread(image_file)
    else:
        raise ValueError("UNKNOWN FILE EXTENSION: {}".format(image_file))

    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = convert_color(image, color_space)
    else:
        feature_image = np.copy(image)

    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
        if vis is True:
            vis_features.append(spatial_features)
    if hist_feat is True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=bins_range)
        file_features.append(hist_features)
        if vis is True:
            vis_features.append(hist_features)
    if hog_feat is True:
        # Call get_hog_features()
        if hog_channel == 'ALL':
            hog_features = []
            hog_vis = []
            for channel in range(feature_image.shape[2]):
                if vis is True:
                    hfeat, hvis = get_hog_features(feature_image[:, :, channel],
                                                   orient, pix_per_cell, cell_per_block,
                                                   vis=vis, feature_vec=True)
                    hog_features.append(hfeat)
                    hog_vis.append(hvis)
                else:
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=vis, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            if vis is True:
                hog_features, hog_vis = \
                    get_hog_features(feature_image[:, :, hog_channel], orient,
                                     pix_per_cell, cell_per_block, vis=vis, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=vis, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
        if vis is True:
            save_hog_sequence('/'.join([vis_folder, "{}-hog-sequence.jpg".format(tag)]),
                              tag, image, hog_vis, np.concatenate(vis_features))

    return np.concatenate(file_features)


def extract_features_list(imgs, color_space='RGB', spatial_size=(32, 32),
                          hist_bins=32, orient=9,
                          pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                          spatial_feat=True, hist_feat=True, hog_feat=True,
                          tag='', vis_count=5, vis_folder=None):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for i, file in enumerate(imgs):
        vis = i < vis_count

        file_features = \
            extract_features(file, color_space=color_space, spatial_size=spatial_size,
                             hist_bins=hist_bins, orient=orient,
                             pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                             spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                             tag="{}-{}".format(tag, i), vis=vis, vis_folder=vis_folder)

        features.append(file_features)
    # Return list of feature vectors
    return features


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, svc, X_scaler, ystart=400, ystop=704, scale=1,
              orient=9, pix_per_cell=4, cell_per_block=2, cells_per_step=2,
              spatial_size=(32, 32), hist_bins=32, grid=False):

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1 or grid is True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

    return bboxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def nonoverlapping_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    # Return the image
    return bboxes


def draw_bboxes(img, bboxes, color=(0, 0, 255), thick=4):
    # Iterate through all detected cars
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img


def produce_heatmap(image, raw_bboxes, threshold=1):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, raw_bboxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    return heatmap


def fuse_bboxes(heatmap):
    labels = label(heatmap)
    fused_bboxes = nonoverlapping_bboxes(labels)
    return fused_bboxes

