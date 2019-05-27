---
layout: post_entry
title: Reflections on Developing Deep Learning Projects
---

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

#### Collect data 

- train, (train-dev), dev, test sets
- data augmentation

After setting up the "barebones neural network", it comes to the step that seems to be the most boring, but in fact a very crucial one - data collection and splitting.

By far, data is the blood of deep learning. Thus, it is essential to understand:
 1. What data the system needs to work on
 2. How to collect relevant data to power such a deep learning system
 
 It is great if a large amount of training data is available "out-of-the-box". However, that is not the case most of the time, so online images and YouTube videos become the common data sources. 


#### Train a network


#### Define evaluation metric

- single value

- benchmark


#### Iterate the training

- human level error (bayes error)
- bias
- variance
- data mis-matching


#### Network Diagnose

- network structure / layer (train certain layers, depends on features)
- learning rate
- optimiser
- loss function
- activation function


#### Other techniques

- transfer learning
- multi-task learning


#### Deployment