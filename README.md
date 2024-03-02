WheatLeaf disease Detection is a condition that presents significant challenges, to public
health globally. It leads to the deterioration of the structure and function of the system.
Accurately predicting diseases at a stage is crucial for managing the risk of disease
progression. In years deep learning has proven to be a tool for predictive modeling in
various medical fields. Utilizing advanced techniques such as VGG19 and
ResNet50, this research paper aims to distinguish between brains affected by WheatLeaf
disease and healthy ones while leveraging extensions to Convolutional Neural Networks
(CNNs), thereby mitigating the reliance solely on traditional CNN architectures. The study
effectively identifies MRI data from individuals with AD. Utilizes known architectures like
VGG19, Resnet50 and others on the trained data to achieve maximum accuracy, in
differentiating between brains affected by AD and those functioning normally.
Keywords; Image Classification, ResNet, VGG16, VGG19 .

 Significance & Challenges
Significance:
The value of this research study depends on resolving the pushing international wellness
obstacle postured by WheatLeaf Disease illness with cutting-edge plus precise very early forecast
approaches. The research concentrates on leveraging deep understanding styles particularly
VGG19 coupled with Resnet50 . By leaving
Convolutional Neural Networks (CNNs), the study intends to add to the growth of much
more effective plus precise analysis devices. Effective application of deep knowing
versions in this context can change the area of neurodegenerative illness medical diagnosis
supplying an unique strategy to enhancing the lifestyle for people impacted by.
1. Enhanced Accuracy as well as Precision :
Utilizing deep understanding versions like VGG19 along with Resnet50 the research study
intends to considerably improve the precision as well as accuracy of WheatLeaf condition
discovery. By using innovative anticipating modeling devices the study makes every effort
to enhance the integrity of analysis results, adding to a lot more precise very early forecasts.
2. Speed and Efficiency :
The fostering of deep knowing styles stresses the relevance of rate and also effectiveness
in the difference procedure. Making use of well-known designs such as VGG19 together
with Resnet50 the research means to improve the evaluation of Magnetic Resonance
Imaging (MRI) information making sure speedy coupled with effective recognition of ADaffected minds.
3. Decrease of Subjectivity :
Deep knowing versions supply a purpose and also standard strategy, reducing the
subjectivity fundamental in typical analysis approaches. By leaving Convolutional Neural
Networks (CNNs) as well as depending on designs like VGG19 as well as Resnet50, the
research study intends to decrease very subjective analyses plus boost the general
dependability of medical diagnoses.
4. Taking Care Of Large Datasets :
Attending to the obstacle of big datasets in clinical imaging research study the research
utilizes deep understanding strategies to effectively handle together with procedure
comprehensive MRI information. This strategy guarantees durable training of the selected
designs enabling extensive evaluation as well as enhanced handling of big datasets.
5. Source Optimization :
Deep discovering's capacity to instantly remove pertinent functions without considerable
5
pre-processing aligns with the objective of source optimization. The research study intends
to maximize this particular, decreasing source demands together with enhancing
computational effectiveness in the evaluation of AD-affected minds.
6. Scalability as well as Accessibility :
The research highlights the scalability as well as schedule of the analysis technique. By
making use of deep knowing designs like VGG19 and also Resnet50 the research study
means to show that the selected designs are scalable to different information dimensions
and also available for more comprehensive applications, adding to their energy in varied
clinical setups.

A. Prediction using ResNet50
Employing pre-trained convolutional neural network that runs ResNet50
architecture[9], employed with a fully trained heavily pre trained on ImageNet's residual
network uses CNNs supports the implementation of a classification model for WheatLeaf Disease. 
The base model from ResNet50 is built upon frozen layers so that the
inferred features can be preserved as the reference model. Our implementation adds
further layers. This includes one that flattens out. Additional layers added here include
flattening, a densely connected layer with rectified-linear unit (ReLU) activation and
final dense layer(Somehow manage to get rid of those words) with SoftMax activation
to classify our images according to the severity of WheatLeaf Disease. The resulting model uses
the Adam optimizer and sparse categorical crossentropy loss . Training model runs across
50 epochs, utilizing a dataset that includes preprocessed MRI image data from many
classes as possible points. Testing the performance of a model via validation of some
subscale datasets can give orientation about the accuracy of the model. This approach
harnesses the depth of ResNet50 for effective feature extraction and classification in
Wheat Leaf disease prediction.
B.Prediction using VGG16
 The VGG16 [10]model, which is a known neural network architecture has been used
to classify WheatLeaf disease by analyzing preprocessed MRI images. The dataset is
divided into training, validation and testing subsets. Goes through preprocessing steps that
take into account image size and batch processing. To retain the learned features the
convolutional layers of the VGG16 model are frozen while using trained weights from
ImageNet. The model then includes an pooling layer followed by dense layers for
extracting features and making predictions. For optimization the model is compiled using
the Adam optimizer with a loss function called crossentropy. During training, which lasts
for ten epochs, performance metrics such, as accuracy and loss are observed on both the
training and validation datasets. Further analysis of a set of test data provides insights,
into the models ability to adapt and perform well in different scenarios. The matrix that
shows confusion and the classification report offer an assessment of how the model
performs across categories. Visualizing the accuracy and loss during training and
9
validation gives an overview of how the model learns over time. The implementation also
includes predictions on a sample MRI image demonstrating how effectively the model
can identify signs of Alzheimer’s disease. Overall using the VGG16 based approach
proves to be a strategy, for classifying Alzheimer’s disease. It utilizes a known CNN
architecture. Incorporates rigorous evaluation techniques.
C.Prediction using VGG19
Using the VGG19 architecture [11] our implemented convolutional neural network
(CNN) aims to identify patterns that indicate Wheatleaf disease, in pre-processed MRI
images. The dataset is carefully organized into training, validation and testing subsets
and undergoes preprocessing to ensure compatibility with the VGG19 model. We
leverage trained weights from ImageNet and incorporate the VGG19 base model keeping
the convolutional layers frozen to retain learned features. Our model design includes an
pooling layer for feature extraction and dense layers for classification. To compile the
model we use the Adam optimizer and sparse categorical crossentropy loss function.
Training is performed over ten epochs, followed by evaluation on a testing dataset to
obtain metrics like accuracy and loss. We further analyse the models performance across
classes of WheatLeaf disease using a confusion matrix and classification report.
Visualizing training and validation accuracy as loss provides valuable insights, into the
learning dynamics of our model. The implementation concludes with predictions on a
sample MRI image, shows that the VGG19 model’s able to predict the presence of
WheatLeaf pathology.

RESULTS AND ANALYSIS
In the pursuit of effective and accurate Wheatleaf disease prediction and our research
examined and compared some deep learning architectures are ResNet50, VGG16,
VGG19, and EfficientNet. In this section, the main goal is to show the parts obtained from
our full-scale experiment; what we did find, supported by numbers. Whether they can
predict well, in general, how else one might take it. More importantly, we also look at the
performance prediction capabilities, generalization and computational efficiency of these
architectures.
The graph shown in Fig.1. illustrates the accuracy patterns of a ResNet50 model during
its training and validation over epochs. On the axis we can see the accuracy values while
the horizontal axis represents the number of training iterations. The blue line shows how
well the model performed on the training dataset indicating that it effectively learned from
and adapted to it. Starting at 70% accuracy it eventually stabilizes at a 95%. In contrast
the orange line represents the models accuracy on a validation dataset. Although it
fluctuates noticeably it follows a trend with peaks, around 90% and 70%, respectively.
 Fig. 1. Graph of Training Accuracy and Validation Accuracy of ResNet50 Model
The graph as shown in Fig. 2. displays training and validation loss curves for a
ResNet50 model in Alzheimer's disease pre- diction. Initially, training loss sharply
decreases as the model learns, while validation loss fluctuates, possibly indicating
overfitting or data variation. As epochs progress, both losses plateau, suggesting
convergence and good generalization. The model appears well-suited, provided validation
11
loss remains close to training loss.
 Fig. 2. Graph of Training loss and Validation loss of ResNet50 Model
The graph shown in Fig.3. we can see a graph that shows the progress of an EfficientNet
model, in predicting WheatLeaf disease. It illustrates how the training and validation
accuracy change over epochs. At the beginning. The training and validation accuracy
increase, indicating learning. As the training continues there are fluctuations in accuracy in
validation. Towards the epochs the two accuracies become closer to each other. The
validation accuracy remains slightly lower than the training accuracy. This suggests that
there is generalization happening. It is important to evaluate further to ensure better
alignment, between training and validation accuracy.

 Fig. 3. Graph of Training Accuracy and Validation Accuracy of EfficientNet Model
The graph as shown in Fig. 4. we can see a graph that represents the training and
validation loss curves, for an EfficientNet model used in predicting WheatLeaf disease
[13]. At the beginning the training loss decreases rapidly. Then stabilizes as more epochs
are completed. This suggests that the model is effectively learning. The validation loss
follows a trend indicating that the model is able to generalize without overfitting. Both
curves converge, which means that the learning process is effective. However towards the
end there is a difference, between the training and validation loss, which may indicate
overfitting and should be given attention. Overall these dynamics of learning in the
EfficientNet model show promise for predicting WheatLeaf disease. To fully. Make
conclusions from this data. It's important to compare these results with other models like
ResNet50, VGG16, and VGG19 in terms of efficiency and effectiveness 
