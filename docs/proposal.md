# Project Proposal
#  Problem Description
Learn and classify speakers' voices. Learn about who is talking in a certain period in some podcast or anime, and check how accurately it can determine the speaker based on the voice in test data.
 
# Data
For data, we plan to use the 50-person audio dataset provided by kaggle. The data will be audio files of 50 people who each spoke for more than 60 minutes.
Further, data converted to wav format, 16KHz, mono channel and is split into 1min chunks. This dataset can be used for speaker recognition kind of problems. 
 
# Methodology/Algorithm
we represent voice/sound data as an image by converting it into  the log-melt spectrum. 
Third, we train a deep learning model using those data. As a deep learning model, we plan to use a CNN(Convolutional Neural Network) based model.
 
# Related Work
Proceedings of 2016 International Conference on Modeling, Simulation and Optimization Technologies and Applications
A methodology for voice classification based on the personalized fundamental frequency estimation
 
# Evaluation Plan: How will you evaluate your results? 
As a multi-class evaluation method, a confusion matrix is used to evaluate with the correct answer rate. We can also use the metrics such as precision, recall, or f1-score for each class. 
Qualitatively, we can evaluate the model by comparing the score to the classification by a human. As another way, dimensionality reduction can be performed and plotted to determine if the audio can be classified. (t-sne, PCA)
