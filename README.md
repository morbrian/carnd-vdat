# Vehicle Detection and Tracking

This is the 5th and final project in the first third of the Udacity Self Driving Car Nanodegree.

The purpose of this project is to identify other vehicles on the road.

## Rubric Points

Original rubric at Udacity: [RubricPoints](https://review.udacity.com/#!/rubrics/513/view)

---
### Writeup

**1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.**

This file is the writeup.

The [demo_pipeline](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L279) function is written to produce most of the sample images
in this write up. It calls the same sequence of functions as our video procesing pipeline,
but exists separately to help generate examples of each step of our process.

The [process_video](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L344) function prepares the pipeline object and starts video processing. 

---
### Histogram of Oriented Gradients (HOG)

**1. Explain how (and identify where in your code) you extracted HOG features from the training images.**

We used the large and small Udacity provided training sets, which consisted of samples identified as 'car' or 'notcar'.

Here is a sample **car** image:

![car_sample][car_sample]

Next is a sample **not car** image:
 
![notcar_sample][notcar_sample]

We leveraged the [extract_features](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/lessons_functions.py#L193) function from the lessons, modified it
to support additional visualization options.  Our [prepare_classifier](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L79) function is
configured to use the best parameters we discovered.

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
to decide if a particular value was improving our results. 

We explored various options, and `LUV` was always best for color and the accuracy score improved by over 
a percentage point when using `ALL` hog channels instead of just one of them. Attempting to use
more orientations than `9` resulted in worse accuracy.

**3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**

We train the classifier using an `LinearSVC` classifier from SciKitLearn in [prepare_classifier](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L79).

We sought out suggestions from Aaron, our Udacity mentor, who recommended including teh sklearn `ExtraTreeClassifier`
in our processing pipeline, and also suggested using the sklearn `Pipeline` object to bundle the classification process.

To provide a larger collection of samples, we also flipped the images left-right so that we would have each image
from either camera angle, which seemed to help match the white car in the video a little more often.

We split the data into training and testing sets, and during development we monitored the accuracy score
on predictions made by the classifier to decide whether it needed additional tuning.

We consistently found we could easily do much better than 99% accuracy, but due to the high number of samples evaluated
during the sliding window search at various scales, the fraction of a percent errors still showed up frequently during
video processing.

For a period of time we had unknowingly had corrupted our data set by including vehicle images in our non-vehicle
dataset, which caused a period of confusion due to poor video results, but we still were seeing 99-100% accuracy during
training, which still seems surprising.

---
### Sliding Window Search

**1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?**

The [find_cars](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/lessons_functions.py#L240) function performs a sliding window search on an image. It uses
and overlapping grid approach to partition the larger image into smaller overlapping images which can be
fed to the classifier to decide if a vehicle is present.

 We chose [three scales](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L55) (0.9, 1.2, 1.8), and for each scale we supply a different
 `window_scale`, `cells_per_step` and start/stop pixels along x and y. 
 
Changing scales helps matching the vehicles at further distances, and we use `cells_per_step=1` at the smallest
scale where the extra granularity is useful, and drop to `cells_per_step=2` at the larger scales.

Each of the next three images shows our search grid for each of the three scales we configured.

![grid_scale09][grid_scale09]
![grid_scale12][grid_scale12]
![grid_scale18][grid_scale18]

**2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?**

Below are two consecutive frame images visualizing how the pipeline identifies bounding boxes around
the vehicles discovered in the image.

Note how the **Frame Heat** column shows more matches, but these are weeded out as noise in the **Historic Heat**
column and the associated bounding boxes are excluded in the final **Fused** image. We chose this specific sequence
because it demonstrates a problem area where the shadowy area is filtered out in early frames, but then manages to
to exceed the threshold briefly, slipping into the fused frame before being filtered out again.

![search_sequence1][search_sequence1]
![search_sequence2][search_sequence2]
![search_sequence3][search_sequence3]
![search_sequence4][search_sequence4]

---
### Video Implementation

**1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding).**

Our final video is at [./output_folder/vdat_project_video.mp4](./output_folder/vdat_project_video.mp4)

**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

We used the processing framework proposed in the lesson to process the video, and then augmented it with
some additional processing to reduce false postitives and to strengthen matches on the vehicles.

The highlevel process applied to each frame is outlined in [apply_pipeline](https://github.com/morbrian/carnd-vdat/blob/c0a74707077c91a2519474f21a91d0d20bd4783f/vdat-pipeline.py#L213), 
and we provide additional detail below to describe how this process works.

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

4. Draw fused bounding boxes

    Finally the fused bounding boxes are drawn on the image frame.
    
---
### Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

Below we list some specific issues we encountered and how we resolved each of them, and we also included a closing
remark for each item to discuss how our solution might fail or could be improved.

1. We initially had trouble matching vehicles consistently even though our classifier had better than 99% accuracy.

    To improve our matching ability, we increased the granularity of the sliding window search at smaller scales
    (ie. stepped by fewer cells), and we searched at 3 different scales instead of just one scale. 
    And we used different cell steps and start/stop parameters at each scale.
    
    
2. We had lots of false positives and lots of missed matches depending on the frame.

    We implemented a bounding box history to remember previous matched locations. This helped maintain a likely 
    location of the vehicle, even when a particular position or lighting condition caused it to be missed
    in a particular frame. It also helped remove false positives, since those were less likely to match
    in every frame.
    
    **Remarks:** Objects which change position quickly over a few frames will still not be matched very well
    with this approach because the history does not account for significant motion or position changes.
    
3. Our classifier remains somewhat sensitive to shadowy areas on the road.

    We were seeing high accuracy scores from our classifier on the training data and test data,
    and although we flipped the images to increase the quantity of data we did not jitter the data
    in any other way. When multiple light and dark areas of the road are scanned, such as when
    light is shining through tree banches onto the road, the classifier is less accurate.
    
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
[grid_scale09]: ./output_folder/grid_scale09_frame00720.jpg
[grid_scale12]: ./output_folder/grid_scale12_frame00720.jpg
[grid_scale18]: ./output_folder/grid_scale18_frame00720.jpg
[search_sequence1]: ./output_folder/search_sequence_00985.jpg
[search_sequence2]: ./output_folder/search_sequence_00986.jpg 
[search_sequence3]: ./output_folder/search_sequence_00987.jpg 
[search_sequence4]: ./output_folder/search_sequence_00988.jpg 
[project_video]: ./output_folder/vdat_project_video.mp4
