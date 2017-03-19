# Vehicle Detection and Tracking

This is the 5th and final project in the first third of the Udacity Self Driving Car Nanodegree.

The purpose of this project is to identify other vehicles on the road.

## Rubric Points

Original rubric at Udacity: [RubricPoints](https://review.udacity.com/#!/rubrics/513/view)

---
### Writeup

**1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.**

This file is the writeup.

The [demo_pipeline](./vdat-pipeline.py) function is written to produce most of the sample images
in this write up. It calls the same sequence of functions as our video procesing pipeline,
but exists separately to help generate examples of each step of our process.

The [process_video](./vdat-pipeline.py) function prepares the pipeline object and starts video processing. 

---
### Histogram of Oriented Gradients (HOG)

**1. Explain how (and identify where in your code) you extracted HOG features from the training images.**

We used the large and small Udacity provided training sets, which consisted of samples identified as 'car' or 'notcar'.

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
![car1f_sequence][car1f_sequence]
![car2_sequence][car2_sequence]
![car2f_sequence][car2f_sequence]
![notcar1_sequence][notcar1_sequence]
![notcar1f_sequence][notcar1f_sequence]
![notcar2_sequence][notcar2_sequence]
![notcar2f_sequence][notcar2f_sequence]

**2. Explain how you settled on your final choice of HOG parameters.**

We tried various values for these parameters, and used the `accuracy_score` prediciton on the classifier
to decide if a particular value was improving our results. We found the `LUV` color space to work best.
We also chose to include `ALL` hog channels, and included both color and spatial histograms as part of
our features array.

**3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**

We train the classifier using an ` LinearSVC` classifier from SciKitLearn in [prepare_classifier](./vdat-pipeline.py).

To provide a larger collection of samples, we also flipped the images left-right so that we would have each image
from either camera angle, which seemed to help match the white car in the video a little more often.

We split the data into training and testing sets, and during development we monitored the accuracy score
on predictions made by the classifier to decide whether it needed additional tuning.

We consistently found we could easily do better than 99% accuracy, but due to the high number of samples evaluated
during the sliding window search at various scales, the fraction of a percent errors still showed up frequently during
video processing.

---
### Sliding Window Search

**1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?**

The [find_cars](./lessons_functions.py) function performs a sliding window search on an image. It uses
and overlapping grid approach to partition the larger image into smaller overlapping images which can be
fed to the classifier to decide if a vehicle is present. For each sub-partition of the grid where a vehicle
is found, the bounding box corners of the partition are stored.

The grid itself covers only the bottom portion of the image where vehicles are likely to appear. The image
below shows an example of how the overlapping grid appears.

![search_grid][search_grid]

**2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?**

Below are two consecutive frame images visualizing how the pipeline identifies bounding boxes around
the vehicles discovered in the image.

Note how the **Frame Heat** column shows more matches, but these are weeded out as noise in the **Historic Heat**
column and the associated bounding boxes are excluded in the final **Fused** image.

![search_sequence1][search_sequence1]
![search_sequence2][search_sequence2]

---
### Video Implementation

**1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding).**

Our final video is at [./output_folder/vdat_project_video.mp4](./output_folder/vdat_project_video.mp4)

**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

We used the processing framework proposed in the lesson to process the video, and then augmented it with
some additional processing to reduce false postitives and to strengthen matches on the vehicles.

The highlevel process applied to each frame is outlined in [apply_pipeline](./vdat-pipeline.py), 
and we provide some additional detail below to describe what this does.

1. Identify raw bounding boxes

    This uses a sliding window search to identify all matches in the image at 3 different scales. It generates
    a heatmap for a single frame, and then combines areas of overlapping matches according to a threshold
    so that weak matches are removed. The fused bounding boxes are remembered in the history.
    
2. Calculate historic heatmap

    This uses a history of recently observed bounding boxes, and combines these matches to create a
    new historic heat map. 
    
3. Identify fused bounding boxes

    The historic heatmap is fused into several new bounding boxes using
    a higher threshold than was used on the single frame fusion. This resulting bounding box
    is what we use to annotate vehicle locations in the image frame.
    
---
### Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

Below we list some specific issues we encountered and how we resolved each of them, and we also included a closing
remark for each item to discuss how our solution might fail or could be improved.

1. We initially had trouble matching vehicles consistently even though our classifier had better than 99% accuracy.

    To improve our matching ability, we increased the granularity of the sliding window search (ie. stepped by fewer cells),
    and we searched at 3 different scales instead of just one scale.
    
    **Remarks:** This helped get matches but also significantly slowed down our per frame processing.
    
2. We had lots of false positives and lots of missed matches depending on the frame.

    We implemented a bounding box history to remember previous matched locations. This helped maintain a likely 
    location of the vehicle, even when a particular position or lighting condition caused it to be missed
    in a particular frame. It also helped remove false positives, since those were less likely to match
    in every frame.
    
    **Remarks:** Objects which change position quickly over a few frames will still not be matched very well
    with this approach because the history does not account for motion or position changes.
    
3. We had a lot of trouble matching the white car and ignoring the left lane line when it is curved.

    We ended up augmenting our training data with flipped versions of every image to help provide examples
    of vehicles from any angle, and this helped with the white car. The left lane line is still frequently
    interpreted as a vehicle. We were able to mostly resolve the left lane line issue by tuning our
    thresholds for both the per-frame boxes and the history bounding boxes.
    
    **Remarks:** The take away lesson we learned from tuning our classifier was that high accuracy of nearly 100%
    may look good, but the small errors show up way more than expected during frame processing where we are
    running the prediction on hundreds of search windows.

[//]: # (Image References)

[car_sample]: ./samples/1.jpeg
[notcar_sample]: ./samples/extra01.jpeg
[car1_sequence]: ./output_folder/car-0-hog-sequence.jpg
[car1f_sequence]: ./output_folder/car-0-flip-hog-sequence.jpg
[car2_sequence]: ./output_folder/car-1-hog-sequence.jpg
[car2f_sequence]: ./output_folder/car-1-flip-hog-sequence.jpg
[notcar1_sequence]: ./output_folder/notcar-0-hog-sequence.jpg
[notcar1f_sequence]: ./output_folder/notcar-0-flip-hog-sequence.jpg
[notcar2_sequence]: ./output_folder/notcar-1-hog-sequence.jpg
[notcar2f_sequence]: ./output_folder/notcar-1-flip-hog-sequence.jpg
[search_grid]: ./output_folder/search_grid.jpg
[search_sequence1]: ./output_folder/search_sequence_frame00309.jpg
[search_sequence2]: ./output_folder/search_sequence_frame00310.jpg 
[project_video]: ./output_folder/vdat_project_video.mp4
