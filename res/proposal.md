- Problem Description
Learn and classify voice actors' voices. Learn the voice actor's voice (such as the voice of the radio), and check how accurate it can be determined by including the voice as test data.
 
- Data
We use the radio videos on Nico Nico Douga and the audio from the Japanese Society of Voice Actors.Download Nico Nico Douga and extract only the audio. The Japan Voice Actors Association can download the audio, so download it.
 
- Methodology/Algorithm
Annotate the data.
Extract features from the data.
Learn with CNN or a Multilayer Perceptron (MLP) deep learning model.
 
- Related Work
Proceedings of 2016 International Conference on Modeling, Simulation and Optimization Technologies and Applications
A methodology for voice classification based on the personalized fundamental frequency estimation
 
 
 
- Evaluation Plan 
As a multi-class evaluation method, a confusion matrix is used to evaluate with the correct answer rate.
The exact match ratio is used for multi-label classification and the labels are evaluated at the exact predicted ratio.
Qualitatively, it is judged by the result of the confusion matrix.
Dimensionality reduction is performed and plotted to determine if the audio can be classified. (t-sne, PCA)
