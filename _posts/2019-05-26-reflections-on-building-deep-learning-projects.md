---
layout: post_entry
title: Reflections on Developing Deep Learning Projects
---

#### Contents

+ [Understand the problem](#understand-the-problem)

+ [Build an initial system](#build-an-initial-system)

+ [Prepare data](#prepare-data)
   
    + [Collect data](#collect-data)
    
    + [Split data](#split-data)
    
    + [Data mis-matching](#data-mismatching)
    
    + [Augment data](#augment-data)

+ [Define evaluation metric](#define-evaluation-metric)

+ [Train a network](#train-a-network)
    
    + [Bias](#bias)
    
    + [Variance](#variance)
    
    + [Data mis-matching](#data-mismatching)

+ [Diagnose network](#diagnose-network)

    + [Error analysis](#error-analysis)
    
    + [Network structure](#network-structure)
    
    + [Convergence](#convergence)

+ [Other techniques](#other-techniques)

+ [Conclusion](#conclusion)


Although I used Artificial Neural Network in my thesis project for my bachelor of engineering, my journey of deep learning in the real world started about three years ago. The problem I was facing was to detect and segment logos from images with plain background. The deep learning solution I used was [Faster R-CNN](https://arxiv.org/abs/1506.01497). Before the project, I didn't know what [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network), [ImageNet](http://www.image-net.org/), [VGG16](https://neurohive.io/en/popular-networks/vgg16/) are, and of course I didn't know what [TensorFlow](https://www.tensorflow.org/) does either. I was extremely lucky to have my then colleague, a very competent Computer Vision Postdoc, as my mentor to help me set foot on my deep learning journey, which I'm forever grateful for.

Since then, I have done various deep learning projects, mainly in computer vision area, including image segmentation, image recognition, object detection and classification, speech detection on digits, and pattern recognition. Recently, an online course ["Structuring Machine Learning Projects" by Andrew Ng](https://www.coursera.org/learn/machine-learning-projects) was mentioned to me, so I decided to check it out and see if it resonates what I have been doing these few years.

Even though the difficulty of the course is listed as beginner level, I found it a very good material to consolidate the steps, approaches and techniques of undertaking a deep learning project, even for engineers with deep learning experience. It actually urged me to gather my thoughts on conducting my previous deep learning projects, and consolidate a clear guideline for my future projects.

<br>

#### Understand the problem 

<br>

When it comes to building a deep learning system, the first thing is to understand the project in hand before getting too deep into the techniques. I would like to ask myself the following questions at the very beginning:

1. Do I really need neural network to solve this problem?

    Neural Network is a very powerful tool to solve complex problem, but it also requires fair amount of data, resources, time and efforts. Additionally, expertise is required to open the black-box, understand and explain what's going on inside. If other machine learning techniques can easily solve the problem, then those approaches are definitely preferred over neural network, because they tend to be more straightforward, and require less resources.

2. Once I decide to go with neural network, which classes/types of neural network should I use?

    Different classes and types of neural networks serve different purposes. 
    For example, Convolutional Neural Network is commonly applied to analyse visual image data (data with hierarchical pattern), while [Recurrent Neural Networks (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) is more suitable for sequential data with its internal state (memory).
    
    If Convolutional Neural Network is preferred, then the understanding goes deeper to which types of CNNs are the right approach.
    For instance, object categorisation requires classification neural network, whereas object detection requires Region Proposal or similar neural networks.

3. Once I decide on the overall techniques, I would see if the problem can be solved with one end-to-end deep learning neural network, or do I need to break it down into multiple stages with different neural networks at each?

    As mentioned in the online course, this depends on the complexity of the problem itself and the amount of data available to train the network. Not all problems can be easily translated as "mapping x to y (`x -> y`)", and some of them may have to be `x -> u -> v -> w -> y`.
    
    Take my previous project of kangaroo detection/recognition as an example. 
    
    An end-to-end deep learning system is the right approach when I just need to detect and classify kangaroos from images of a certain environment. Because a deep learning network is sufficient to handle tasks of object detection and classification. Also, there are plenty of kangaroo images on the Internet available for training such a network.
    
    However, if I want to not only detect the kangaroos in the images, but also recognise their activities, such as moving left or right, and grazing or not, the problem becomes more complicated. It requires much more information for the neural network to analyse and learn than just kangaroo detection/classification. On the other hand, there might not be that easy to collect a large amount of labelled data for each activity.
      
    In this case, a multi-stage system is more appropriate. The first step would be to detect where the kangaroos are in an image, and then crop / extract the object out. The second step would be to detect where the head and tail of the kangaroos are in the cropped image, using a cascaded object-detection neural network. The final step would be to use heuristic methods to decide on the activities of the kangaroos.

<br>

#### Build an initial system

<br>

Once the problem is understood thoroughly, building an initial system would be the next step. 

Since there are many open-source implementations and libraries available for most of the commonly-used neural networks, it wouldn't be too hard to focus on the suitable ones and modify the network structure accordingly if needed.

I personally would like to set up and run the initial system quickly (which is also suggested by Andrew Ng), with minimum configuration and sample data. To me, this is the most efficient way to understand the overall structure of the system, the working mechanism of the network and the required format of the input data. Those are the main areas that I found useful to sophisticate the system in the later steps.

When building a deep learning system, another important aspect is what the course called "Orthogonalization". As hyperparameter tuning is almost inevitable in developing machine learning systems, we want to be able to change / configure one parameter at a time and compare the system performance, which is similar to [A/B testing](https://en.wikipedia.org/wiki/A/B_testing) in software development. Therefore, flexible configuration in the system can significantly increase efficiency in the process of training network.

<br>

#### Prepare data 

<br>

After setting up the "barebones neural network", it comes to the step that seems to be the most boring, but in fact a very crucial one - data collection and splitting.

<br>

<a name="collect-data"></a>**Collect data**

<br>

By far, data is the blood of deep learning. Thus, it is essential to understand:

1. What data the system needs to work on

2. How to collect relevant data to power such a deep learning system
 
It is great if a large amount of training data is available "out-of-the-box". However, that is not the case most of the time, so online images and YouTube videos become the common data sources. 
 
<br>

<a name="split-data"></a>**Split data**

<br>
 
Once the data collection step has been completed, the following step is to split the data for training, development and testing. 
 
+ Training set - data set that is used to train the neural network

+ Development set (dev set) - data set that is used to tune the trained neural network based on its accuracy and performance

+ Test set - data set that is used to evaluate the performance of the trained neural network
 
As mentioned in the course, there are mainly two different scenarios, depending on the size of the collected data (I would use 100,000 as a cut-off threshold), when it comes to splitting the data:
 
1. Relative small amount of data collection 
 
    If the size of the collected data is less than 100,000, then a traditional rule-of-thumb for data splitting can be applied: 70% for training and 30% for testing. Inside the 30% test set, data can be further split into 15% for evaluation (development) and 15% for actual testing.
 
2. Relative large amount of data collection 
 
    If the size of the collected data is more than 100,000, then a more appropriate rule-of-thumb is 98%
for training, 1% for development and 1% for testing. Since the size of the data collection is quite large, 1% of data should be sufficient for development or evaluation. Additionally, this allows more data for training data-thirsty neural networks to achieve a better accuracy.

The above is based on the assumption that the data distribution between training set and dev/test set are the same. That is, the data for training and the data that the network needs to perform prediction on have the same or similar attributes (resolution and quality etc.). It is probably not realistic to expect a neural network trained on high-resolution images to perform classification well on low-resolution and blurry images.
 
<br>
 
<a name="data-mismatching"></a>**Data mismatching**
 
<br>
 
If there is a mis-match between the training data and the target data (the actual data that the network will work with in production), then we may want to treat the training, development and test data sets slightly differently.
 
1. The development and test data sets should come from the same distribution, and that distribution should be similar to the target data.
 
2. The training data may be from a different distribution/source from the target data, but it should also contain a small amount of data that comes from the same distribution as the target data. 
 
3. To better understand the neural network performance, it may be worth having another data set called "train-dev", along with the previous training, dev and test sets. The train-dev set contains data that has same distribution as the training set, but different from that of dev/test sets. 
 
This train-dev set is used for development only, instead of training. The purpose of it is to better understand the network performance and help diagnose the trained neural network, which will be discussed later in the article.
 
<br>
 
<a name="augment-data"></a>**Augment data**

<br>

Apart from data collection and splitting, data augmentation is another important step in data preparation for training a neural network. Data augmentation is to generate altered copies of the existing data in the training set.

Personally I think the two main reasons for data augmentation are as following:

1. To generate more training data under the circumstances where lack of training data is a problem

    Take image data as an example, augmentation on image data include but is not limited to:

    + Add gaussian noise
    
    + Inverse images
   
    + Blur images
    
    + Add random rotation, shifts, shear and flips

2. To generate training data that are close to the target data when there is a data mis-matching issue

    For example, if the target data is known to be slightly blurry, a simple and straight-forward way to augment the training data is to artificially make them blurry so that they are similar to the target data.

<br>

#### Define evaluation metric

<br>

Before getting too deep into iterating the training of a neural network, it's always a good idea to define an evaluation metric and a benchmarking process. This way, benchmark can be run against the trained network at each iteration, and evaluation metrics can be compared among different versions of trained networks to determine if the performance is getting better or worse.

To evaluate the performance of a trained neural network, I mostly use the following two measures:

1. [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

2. Accuracy or [F1-score](https://en.wikipedia.org/wiki/F1_score)

Confusion matrix gives a clear view of the neural network performance across all classes. It not only shows the accuracy of each class, but also shows how each class is mis-classified into other classes. This gives insights on which class(es) to focus on in order to improve the overall accuracy of the neural network.

On the other hand, a single value metric, such as accuracy or F1-score, provides an overall measurement on how well the neural network performs, which is useful and efficient to compare performances across different iterations of trained networks. F1-score is considered to be a balanced measurement, as it takes both [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) into account. It not only examines if the neural network mis-classify data into incorrect classes (false positive), but also considers if the network is unable to classify data into certain classes (false negative).

<br>

#### Train a network

<br>

When training a neural network, I usually first focus on the indicator of the completion of training - the convergence of loss. Only when the loss of the neural network converges during the training process, can we say the training is completed and the accuracies of both training and valuation can be trusted.

To understand how well the trained neural network is, as mentioned in the course, I would focus on several numbers:

+ Human level error (Bayes error)

  [Bayes error rate](https://en.wikipedia.org/wiki/Bayes_error_rate) is the lowest possible error rate of a classifier. Since human are extremely good at problems with natural perceptions, such as image classification and natural language processing, for those problems, human level error is close to Bayes error; therefore, they can be interchangeable.
  
+ Training error

  The final error rate of training the neural network.
  
+ Dev error

  The error rate when applying the trained neural network to the dev data set.
  
+ Test error

  The error rate when applying the trained neural network to the test data set.

These numbers are good indicators of the general problem of a neural network, such as high bias, high variance or data mis-matching.

<br>

<a name="bias"></a>**Bias**

<br>

In short, high [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator) suggests that the neural network is not able to correlate the relations between data features and the corresponding output. This is also interpreted as underfitting.

Take the following as an example:

<br>

{:class="table table-bordered"}
| Error type        | Error rate | Difference |
|:-----------------:|:----------:|:----------:|
| Human level error | 0.1%       | n/a        | 
| Training error    | 5.1%       | 5%         |
| Dev error         | 5.2%       | 0.1%       |
| Test error        | 5.5%       | 0.3%       |

<br>

The difference between training error and human level error is 5%, much larger than that between dev error and training error or that between test error and dev error. This indicates that the training of the neural network is not able to correctly map features in the data to the expected outputs.

To adjust this issue, the followings can be considered:

+ Use a larger / more complex neural network

+ Train longer / with better optimisation algorithms (e.g. add momentum, RMS prop, Adam)

+ Neural network architecture / hyper-parameters search

<br>

<a name="variance"></a>**Variance**

<br>

In contrast to bias, high [variance](https://en.wikipedia.org/wiki/Variance) suggests that the neural network is too sensitive to small changes in the training data; thus, it is not able to provide a generalised model. This is also interpreted as overfitting.

For instance:

<br>

{:class="table table-bordered"}
| Error type           |  Error rate   |  Difference   |
|:--------------------:|:-------------:|:-------------:|
| Human level error    | 0.1%          | n/a           | 
| Training error       | 0.3%          | 0.2%          |
| Dev error            | 5.3%          | 5%            |
| Test error           | 5.5%          | 0.2%          |

<br>

The difference between training error and human level error is quite small (0.2%); however, the difference between dev error and training error is 5%, which is relative large in comparison. This implies that the training process overfit the training data and is unable to generalise the model to achieve a similar accuracy on the dev data set.

The followings are common methods to avoid overfitting:

+ Increase training data set with data augmentation

+ Add regularisation (e.g. L2, dropout)

+ Reduce the complexity of the neural network

+ Neural network architecture / hyper-parameters search

<br>

<a name="data-mismatching"></a>**Data mismatching**

<br>

Apart from avoidable bias and variance, there is another possible issue affecting the performance of the neural network - data mismatching between training set and target set, which is more difficult to tackle.

For example:

<br>

{:class="table table-bordered"}
| Error type           |  Error rate   |  Difference   |
|:--------------------:|:-------------:|:-------------:|
| Human level error    | 0.8%          | n/a           | 
| Training error       | 1%            | 0.2%          |
| Training-dev error   | 1.5%          | 0.5%          |
| Dev error            | 9%            | 7.5%          |
| Test error           | 9.5%          | 0.5%          |

<br>

In the table above, the difference between training error and human level error is 0.2%, while the difference between training-dev error and training error is 0.5%, both of which are relatively small. Therefore, bias and variance don't seem to be a big issue of this neural network. However, the difference between dev error and training-dev error is relative large (7.5%), that indicates there is mismatched data between the training set and the target set.

This is a tricky problem as it is not easy to address it in a systematic way. One way to address the issue is to manually identify the differences between the training and target data. Then we can try to collect or generate training data that is more similar to the target data (dev/test sets). This technique is sometimes referred as artificial data synthesis.

For example, as mentioned in the course, if the target images are mostly foggy images whereas those in the training set are images under clear weather. To make training data more similar to the target ones, we can artificially synthesise new training data by combining the existing images in the training set with some random images of fogs. 

One caveat to keep in mind when undertaking this approach is that synthesised data might cause the trained neural network to overfit data with repetitive noise, such as the fog images in the example mentioned previously.

<br>

#### Diagnose network

<br>

In my experience, there was barely a time that a neural network is "ready-to-use" after the first training. The reality is that the performance of the first trained neural network is suboptimal, and diagnosis is required to further understand the issue in order to achieve improvements.

The followings are the areas that I have experienced to be effective in achieving better performance of the neural networks.

<br>

<a name="error-analysis"></a>**Error analysis**

<br>

First thing I would look at when diagnosing a trained network is the data. Mislabeled data in dev data set can not only be easily fixed but also affect the performance of the trained network significantly.

One way to do so is to first check if there is mislabeled data in the dev set. If so, then to sample a small amount of mislabeled data and look through them to see how the data is mislabeled. Once the issues among the data is understood, fixes can be applied relatively easily.

There is also a chance that mislabeled data exists in the training set. However, since the size of the training data is much larger than dev set in most cases, it would be a time-consuming task to go through the training set and fix the issue. As long as the mislabeled training data does't happen in a systematic way (i.e. mislabeled data is random and rare), they can be ignored.

<br>

<a name="network-structure"></a>**Network structure**

<br>

If the data is clean and accurate, the other potential factor that has a large impact on the performance is the neural network structure itself. Especially for large complex network, not every layer in the network needs to be trained. 

The "shallow" layers (first `x` layers in the network) are trained to capture low-level features, while the "deeper" layers (last `x` layers) are trained to capture features in higher levels. 

+ If the training data set is not huge or the features are fairly common with the open-source data sets, then it may be a good idea to use the open-sourced pre-trained weights, fix the "shallow" layers (disable them in training as `non-trainable`), and only train the "deeper" layers to capture high-level features of the data set ([transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)).

+ If the training data set is large (100,000+ data points) or the features are different from the open-source data set, then it is probably worth training the network from a "shallow" layer all the way to the last layer.

<br>

<a name="convergence"></a>**Convergence**

<br>

A strong indicator of how the training process performances is the convergence of the loss (or the accuracy). If the loss of the network is decreasing during the training and gradually converging to a certain value (ideally zero), then the training process is as expected.

However, if the loss doesn't converge, in the case of backpropagation and gradient descent, it could be because local minimums or maximums are hit during the gradient descent process, reducing learning rate can be an effective fix to this problem.

In other cases, modifications of loss function and activation function can be experimented. 

For example, 

+ [sigmoid function](https://en.wikipedia.org/wiki/Logistic_function) is commonly used for binary classifiers, 

+ [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is commonly used for multi-class classifiers,

+ [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))is very popular among deep neural networks, due to its fast speed, sparse activation, and better gradient propagation (fewer [vanishing gradient problems](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

As for loss functions, for instance, the losses for object detection and recognition are coupled in the loss function in some two-phase convolutional neural networks, which might affect the overall performance. Decoupling them could potentially help to improve the overall accuracy.

<br>

#### Other techniques

<br>

[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is very popular among training and development of deep learning neural networks. However, there is another technique that can be useful in certain circumstances, which is [Multi-task learning](https://en.wikipedia.org/wiki/Multi-task_learning).

Essentially, multi-task learning is to solve multiple learning tasks at the same time. By learning several tasks jointly and exploring the commonalities as well as differences among them, it can result in better learning efficiency and prediction accuracy. 

As mentioned in the course, one advantage of using multi-task learning is to compromise the lack of data for a particular class by learning multiple related classes at the same time. A very simple example is that, when training a traffic-light recognition system, statistically speaking, we should have much more image data of green and red lights than that of orange light. By doing the training with data of green, red and orange lights all together, it will achieve a better recognition accuracy overall than by training them separately, due to the lack of data for orange light. 

<br>

#### Conclusion

<br>

Deep learning is a fast growing domain that requires a lot of deep-dives to gain expertise. The techniques mentioned in this article is just a scratch on the surface, but I hope it's helpful to get a deep learning project off the ground. It is a summary of the problems and solutions that I have experienced and I found worth noting for future reference. That being said, I will keep exploring the domain and expanding the knowledge of deep learning.