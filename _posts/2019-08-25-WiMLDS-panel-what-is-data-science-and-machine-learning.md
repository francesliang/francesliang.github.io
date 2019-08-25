---
layout: post_entry
title: WiMLDS Panel - What is data science and machine learning
---

Recently, I was fortunate enough to be invited as a panelist for Women in Machine Learning and Data Science Melbourne chapter as part of the National Science Week. The discussion topic of the night was "What is data science and machine learning".

It was a great event and some of the questions I prepared for the panel made me self-reflect in depth on the journey of becoming a machine learning engineer. I thought it might be worth sharing. 

<br>

#### Q: What do you love about machine learning (ML)?

<br>

On a personal level, I love maths and science growing up and I also love building things and seeing them working for real. I think machine learning brings the beauty of maths to the real world in a very practical way, and provides concrete solutions to some real problems.

In a bigger picture, I do agree with the viewpoint in a popular book “Sapiens - A brief history of humankind” that we’ve already had lots of huge breakthroughs in our history, and the next one might not only require our biological intelligence but also artificial assistance; machine learning seems to be on the right track of this. I believe it has a bright future.

<br>

#### Q: What are the biggest opportunities for machine learning in Australia and what do you want to build in the future?

<br>

Machine learning still seems to be in their early stage of adoption in Australia, but it’s getting lots of attention, especially in some more traditional industries (resources, infrastructure, regulatory tech, insurance etc.). People are talking about it, are trying to figure out how to apply it to concrete problems but still don't fully understand what it actually is. Therefore, there are lots of disruptive opportunities there.

On the other side, Internet of Things (IoT) / edge computing is another big area for machine learning opportunity, it’s also where my personal interest is. This is because deployed sensors can:
 
 1. generate a huge volume of data in a matter of secs since its data collection is continuously active in real-time;
 
 2. capture a large variety of data as different sensors monitor different interactions / behaviours in the physical environment, such as thermometers to capture temperature and cameras to capture images/videos.
 
The data collected from sensors deployed in various environments are noisy and unorganised. Machine learning techniques would be very useful to derive insightful patterns and indications from these data.

<br>

#### Q: What, if so, were the biggest challenges that you faced along the way developing your career? 

<br>

I would very much want to blame my appearance - giving people an impression that I’m in-experienced and I have to prove myself extra hard. But, the real big challenge is how to turn self-doubt / imposter syndrome into positive energy for learning skills that allow you to pursue what you are truly passionate about; be able to not lose confidence and courage when facing doubt and uncertainty due to both internal and external factors.

<br>

#### Q: Any words of wisdom for Data Science/ML students or practitioners starting out?

<br>

Get your hands dirty - think of or find a machine-learning related problem that you are interested in, and then try to solve it by looking for related data-set (either open-source available online or collect it yourself), and coding / modifying some algorithms.

Along this way, you will come across lots of “what is this / how do I do this” moments, which not only allows you to apply your existing skills, but also urges you to follow your thought process and to look for / learn new knowledge / techniques that you haven’t been exposed before.

The learning curve of solving a practical problem might be much more steep compared to doing online courses, but you will have expanded your technical knowledge as well as gained some practical experience, which are extremely important when applying machine learning to real-world problems.

<br>

#### Q: What do you think makes a good machine learning engineer?

<br>

+ Firstly, you have the passion and curiosity in the data world. If you couldn't find the beauty in the data world, it can ge dreadful and boring.

+ Software engineering skills, including code structure, testing, debugging, good general practice etc.

+ A good intuition and understanding of data and be good at identifying patterns from data.

+ A good sense of metrics, as evaluation/benchmarking an important step in model development. 
    
    Especially for iterative model development, it's essential to define a set of evaluation measures first, then benchmark the performance of a trained model in each iteration using the pre-defined metrics to ensure the model tuning is on the right track.

+ A good understanding of system architecture, since there are lots of moving parts in a ML project, from data input, to iterative process of algorithm/model development, to serving insightful outcomes.

+ A good understanding of the algorithms. If you are going to teach the machine learn something, you should properly know that thing well beforehand, and nothing should be a total black-box in the system.

<br>

#### Q: What will you say the “best practices” in machine learning?

<br>

+ Package the system into a self-contained environment (such as using docker) to capture system specific machine learning/deep learning frameworks and dependencies, so that the platform details are disentangled from your running system.

+ Pushing training and evaluation to the cloud, with an instance that is always up and running to allow long period of training / evaluation.

+ Data management: 

    + automated ETL/data processing step to generate clean data that is machine-learning ready
    
    + version control on data so it's clear which model trained on which data

+ Configuration management: 

    + effective hyper-parameter tuning, with functionality to be able to change one parameter at a time and assess the impact of it on the model performance 
    
    + version control so it's clear which model trained on which set of configs, with which version of the source code

+ Automated testing on model inference, and take evaluation metrics into consideration.

    Testing often gets neglected in many machine learning projects, because of rapid iterations of model training. However, that should be the exact reason why testing needs to be in place. Test suit makes sure that the machine learning system works as expected using updated models. With the consideration of evaluation metrics in the test suit, performance/accuracy of updated models can also be assured.

<br>

#### Q: As someone just starting out I want to showcase my skills what is the best way to showcase that to potential employers/collaborators?

<br>

Build something and write a blog post or make a demo video about it. 

You may hear true stories like someone really wants to work for a company that uses AI / ML to alert grocery stores when it’s time for them to order new inventory. So this person decided to build a basic model to serve this purpose, by going to multiple supermarkets, walking down and recording the aisles many times with their camera to collect data; then they built a machine learning model to identify empty spot in the grocery shelves; also make the application public. This eventually attracted the attention of the company’s CEO.

I think the best way to showcase your skills is to build an application for a practical use case, and package it for demo.
