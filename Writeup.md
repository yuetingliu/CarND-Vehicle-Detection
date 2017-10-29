
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_feature.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/car_detection.png
[image5]: ./output_images/car_detection_heatmap.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the 'LUV' color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally chose the following:
color_space='LUV', orientations=9, pixels_per_cell=(8, 8) and cells_per_block=(2, 2)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM using only HOG features in 'LUV' color space. I firstly tried to use gray scale, and the accuracy was quite high. However, in the video detection, the classifier did not work out well. Then I used 'LUV' color space to extract HOG feature in each channel and stack them together. The feature got longer, but the classier worked out better. I found out the color and spatial info was not very helpful, and they increased the feature length therefore, training time. I decided to use only HOG features in 'LUV' color space

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried many times with different window scales. Eventually I used window size instead of scales, because it was more intuitive and easier to improve. I tried window size from 32, 64, 96, 128, to 512 (corresponding scale = window_size/default_window_size, default_window_size was set as 64). After many trials and errors, I chose window sizes of 64, 96, 128, 160, 255, 300. To improve speed, each window size has its own searching area. Because a window size of 64 would never detect a car near the bottom of the image (unless an extremely small car). I found overlap window of 75% to be suitable.

I remove the left half of the image to improve speed, otherwise it takes forever for my laptop(even my AWS ec2) to search the whole area.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on various of scales using LUV 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

This project turned out very difficult. The first difficulty I met was to get the pipeline working. A small detail could ruin the whole pipeline, for instance, the data range of training samples, png file read by matplotlib.image, was (0,1). However, the vidoe feed gave a data range of (0,255). Another difficulty was about speed. I had to reduce the detection areas for each search window very hard, because searching larger area took too long (10 hours in my first attempt). I printed the time usage of each stage and found out the number of searching steps to be the most determining factor. To make a tradeoff between quality and time, my final result video was not in high quality in terms of car detection. Some frames the car was off the radar. 

I understand deeply why C++ is needed in self-driving car!!! I will keep improving my pipeline and I can think of a few directions. First one is to use neural network for feature extraction or classifier, or both, as some fellow students also pointed out in the slack channel. For car detection, I have seen some projects that use neural network to achieve insanely high speed. For instance, this paper [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf).

This is the final project of term 1, and interstingly, tomorrow is the deadline of passing all the projects. I hope to get passed. But for sure, I will keep improving the quality and keep learning term 2 and 3 for my new career



```python

```