# Vehicle Detection and Tracking

This is the 5th and final project in the first third of the Udacity Self Driving Car Nanodegree.

The purpose of this project is to identify other vehicles on the road.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Writeup

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This file is the writeup.

The [demo_pipeline](./vdat-pipeline.py) function is written to produce most of the sample images
in this write up. It calls the same sequence of functions as our video procesing pipeline,
but exists separately to help generate examples of each step of our process.

The [process_video](./vdat-pipeline.py) function prepares the pipeline object and starts video processing. 

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

We used the large and small Udacity provided training sets, which consisted of samples identified as 'car' or 'notcar'.

There were more 

Here is a sample **car** image:

![car_sample][car_sample]

Next is a sample **not car** image:
 
![notcar_sample][notcar_sample]

We leveraged version of the [extract_features](./lessons_functions.py) function from the lessons, modified
to support additional visualization options.  Our [prepare_classifier](./vdat-pipeline.py) function is
configured to use the best parameters we discovered by default.

Below we use colorspace `LUV`, `orientations=9`, `pix_per_cell=8`, `cells_per_block=2`, `hog_channel=ALL`,
to generate the HOG visualizations, and the plot at the far right includes the color and spatial bins.

![car1_sequence][car1_sequence]
![car2_sequence][car2_sequence]
![notcar1_sequence][notcar1_sequence]
![notcar2_sequence][notcar2_sequence]

####2. Explain how you settled on your final choice of HOG parameters.

We tried various values for these parameters, and used the `accuracy_score` prediciton on the classifier
to decide if a particular value was improving our results. We found the `LUV` color space to work best.
We also chose to include `ALL` hog channels, and included both color and spatial histograms as part of
our features array.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

We train the classifier using an ` LinearSVC` classifier from SciKitLearn in [prepare_classifier](./vdat-pipeline.py).

We split the data into training and testing sets, and during development we monitored the accuracy score
on predictions made by the classifier to decide whether it needed additional tuning.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The [find_cars](./lessons_functions.py) function performs a sliding window search on an image. It uses
and overlapping grid approach to partition the larger image into smaller overlapping images which can be
fed to the classifier to decide if a vehicle is present. For each sub-partition of the grid where a vehicle
is found, the bounding box corners of the partition are stored.

The grid itself covers only the bottom portion of the image where vehicles are likely to appear. The image
below shows an example of how the overlapping grid appears.

![search_grid][search_grid]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are some sample images and frames visualizing how the pipeline identifies bounding boxes around
the vehicles discovered in the image.

![search_sequence1][search_sequence1]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding 

TODO

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

TODO

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO

[//]: # (Image References)

[car_sample]: ./samples/1.jpeg
[notcar_sample]: ./samples/extra01.jpeg
[car1_sequence]: ./output_folder/car-0-hog-sequence.jpg
[car2_sequence]: ./output_folder/car-1-hog-sequence.jpg
[notcar1_sequence]: ./output_folder/notcar-0-hog-sequence.jpg
[notcar2_sequence]: ./output_folder/notcar-1-hog-sequence.jpg
[search_grid]: ./output_folder/search_grid.jpg
[search_sequence1]: ./output_folder/search_sequence.jpg
