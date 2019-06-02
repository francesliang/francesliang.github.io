---
layout: post_entry
title: Reflections on Developing Deep Learning Projects
---

### Contents

* [Understand the problem](#Understand the problem)
* [Build an initial system](#Build an initial system)
* [Prepare data](#Prepare data)
    - [Collect data](#Collect data)
    - [Split data](#Split data)
    - [Data mis-matching](#Data mis-matching)
    - [Augment data](#Augment data)
* [Define evaluation metric](#Define evaluation metric)
* [Train a network](#Train a network)
* [Diagnose network](#Diagnose network)
* [Other techniques](#Other techniques)


Although I used Artificial Neural Network in my thesis project for my bachelor of engineering, my journey of deep learning in the real world started about three years ago. The problem I was facing was to detect and segment logos from images with plain background. The deep learning solution I used was [Faster R-CNN](https://arxiv.org/abs/1506.01497). Before the project, I didn't know what [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network), [ImageNet](http://www.image-net.org/), [VGG16](https://neurohive.io/en/popular-networks/vgg16/) are, and of course I didn't know what [TensorFlow](https://www.tensorflow.org/) does either. I was extremely lucky to have my then colleague, a very competent Computer Vision Postdoc, as my mentor to help me set foot on my deep learning journey, which I'm forever grateful for.

Since then, I have done various deep learning projects, mainly in computer vision area, including image segmentation, image recognition, object detection and classification, speech detection on digits, and pattern recognition. Recently, an online course ["Structuring Machine Learning Projects" by Andrew Ng](https://www.coursera.org/learn/machine-learning-projects) was mentioned to me, so I decided to check it out and see if it resonates what I have been doing these few years.

Even though the difficulty of the course is listed as beginner level, I found it a very good material to consolidate the steps, approaches and techniques of undertaking a deep learning project, even for engineers with deep learning experience. It actually urged me to gather my thoughts on conducting my previous deep learning projects, and consolidate a clear guideline for my future projects.


#### Understand the problem 

When it comes to building a deep learning system, the first thing is to understand the project in hand before getting too deep into the techniques. I would like to ask myself the following questions at the very beginning:

1. Do I really need neural network to solve this problem?

Neural Network is a very powerful tool to solve complex problem, but it also requires fair amount of data, resources, time and efforts. Additionally, expertise is required to open the black-box, understand and explain what's going on inside. If other machine learning techniques can easily solve the problem, then those approaches are definitely preferred over neural network, because they tend to be more straightforward, and require less resources.

2. Once I decide to go with neural network, which classes/types of neural network should I use?

Different classes and types of neural networks serve different purposes. 
For example, Convolutional Neural Network is commonly applied to analyse visual image data (data with hierarchical pattern), while [Recurrent Neural Networks (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) is more suitable for sequential data with its internal state (memory).

If Convolutional Neural Network is preferred, then the understanding goes deeper to which types of CNNs are the right approach.
For instance, object categorisation requires classification neural network, whereas object detection requires Region Proposal or similar neural networks.

3. Once I decide on the overall techniques, I would see if the problem can be solved with one end-to-end deep learning neural network, or do I need to break it down into multiple stages with different neural networks at each?

As mentioned in the online course, this depends on the complexity of the problem itself and the amount of data available to train the network. Not all problems can be easily translated as "mapping x to y (`x -> y`)", and some of them may have to be 'x -> u -> v -> w -> y'.

Take my previous project of kangaroo detection/recognition as an example. 

An end-to-end deep learning system is the right approach when I just need to detect and classify kangaroos from images of a certain environment. Because a deep learning network is sufficient to handle tasks of object detection and classification. Also, there are plenty of kangaroo images on the Internet available for training such a network.

However, if I want to not only detect the kangaroos in the images, but also recognise their activities, such as moving left or right, and grazing or not, the problem becomes more complicated. It requires much more information for the neural network to analyse and learn than just kangaroo detection/classification. On the other hand, there might not be that easy to collect a large amount of labelled data for each activity.
  
In this case, a multi-stage system is more appropriate. The first step would be to detect where the kangaroos are in an image, and then crop / extract the object out. The second step would be to detect where the head and tail of the kangaroos are in the cropped image, using a cascaded object-detection neural network. The final step would be to use heuristic methods to decide on the activities of the kangaroos.


#### Build an initial system

Once the problem is understood thoroughly, building an initial system would be the next step. 

Since there are many open-source implementations and libraries available for most of the commonly-used neural networks, it wouldn't be too hard to focus on the suitable ones and modify the network structure accordingly if needed.

I personally would like to set up and run the initial system quickly (which is also suggested by Andrew Ng), with minimum configuration and sample data. To me, this is the most efficient way to understand the overall structure of the system, the working mechanism of the network and the required format of the input data. Those are the main areas that I found useful to sophisticate the system in the later steps.

When building a deep learning system, another important aspect is what the course called "Orthogonalization". As hyperparameter tuning is almost inevitable in developing machine learning systems, we want to be able to change / configure one parameter at a time and compare the system performance, which is similar to [A/B testing](https://en.wikipedia.org/wiki/A/B_testing) in software development. Therefore, flexible configuration in the system can significantly increase efficiency in the process of training network.

#### Prepare data 

- train, (train-dev), dev, test sets
- data augmentation

After setting up the "barebones neural network", it comes to the step that seems to be the most boring, but in fact a very crucial one - data collection and splitting.

##### Collect data 

By far, data is the blood of deep learning. Thus, it is essential to understand:
 1. What data the system needs to work on
 2. How to collect relevant data to power such a deep learning system
 
 It is great if a large amount of training data is available "out-of-the-box". However, that is not the case most of the time, so online images and YouTube videos become the common data sources. 
 
##### Split data 
 
 Once the data collection step has been completed, the following step is to split the data for training, development and testing. 
 
 - Training set - data set that is used to train the neural network
 - Development set (dev set) - data set that is used to tune the trained neural network based on its accuracy and performance
 - Test set - data set that is used to evaluate the performance of the trained neural network
 
 As mentioned in the course, there are mainly two different scenarios, depending on the size of the collected data (I would use 100,000 as a cut-off threshold), when it comes to splitting the data:
 1. Relative small amount of data collection 
 
 If the size of the collected data is less than 100,000, then a traditional rule-of-thumb for data splitting can be applied: 70% for training and 30% for testing. Inside the 30% test set, data can be further split into 15% for evaluation (development) and 15% for actual testing.
 
 2. Relative large amount of data collection 
 
 If the size of the collected data is more than 100,000, then a more appropriate rule-of-thumb is 98%
for training, 1% for development and 1% for testing. Since the size of the data collection is quite large, 1% of data should be sufficient for development or evaluation. Additionally, this allows more data for training data-thirsty neural networks to achieve a better accuracy.

 The above is based on the assumption that the data distribution between training set and dev/test set are the same. That is, the data for training and the data that the network needs to perform prediction on have the same or similar attributes (resolution and quality etc.). It is probably not realistic to expect a neural network trained on high-resolution images to perform classification well on low-resolution and blurry images.
 
 ##### Data mis-matching
 
 If there is a mis-match between the training data and the target data (the actual data that the network will work with in production), then we may want to treat the training, development and test data sets slightly differently.
 
 1. The development and test data sets should come from the same distribution, and that distribution should be similar to the target data.
 
 2. The training data may be from a different distribution/source from the target data, but it should also contain a small amount of data that comes from the same distribution as the target data. 
 
 3. To better understand the neural network performance, it may be worth having another data set called "train-dev", along with the previous training, dev and test sets. The train-dev set contains data that has same distribution as the training set, but different from that of dev/test sets. 
 
 This train-dev set is used for development only, instead of training. The purpose of it is to better understand the network performance and help diagnose the trained neural network, which will be discussed later in the article.
 
##### Augment data 

Apart from data collection and splitting, data augmentation is another important step in data preparation for training a neural network. Data augmentation is to generate altered copies of the existing data in the training set.

Personally I think the two main reasons for data augmentation are as following:

1. To generate more training data under the circumstances where lack of training data is a problem

Take image data as an example, augmentation on image data include but is not limited to:

- Add gaussian noise
- Inverse images
- Blur images
- Add random rotation, shifts, shear and flips

2. To generate training data that are close to the target data when there is a data mis-matching issue

For example, if the target data is known to be slightly blurry, a simple and straight-forward way to augment the training data is to artificially make them blurry so that they are similar to the target data.

#### Define evaluation metric

- single value

- benchmark


#### Train a network (Iterate the training)

- human level error (bayes error)
- bias
- variance
- data mis-matching


#### Diagnose network

- network structure / layer (train certain layers, depends on features)
- learning rate
- optimiser
- loss function
- activation function


#### Other techniques

- transfer learning
- multi-task learning


#### Deployment