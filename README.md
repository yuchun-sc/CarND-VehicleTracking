# CarND-VehicleTracking
Vehicle detection and tracking

Simple two step vehicle dection pipleline
## 1. Training a binary classifier by using the following feature
1. Bin spatial
2. Histogram
3. HOG

## 2. Apply the classifier over sliding windows
1. generate a series of sliding windows with many sizes and overlapping
2. apply the classifer to each of them 
3. use heat map to find the bounding box with highest confidence of containing a car 
