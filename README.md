This repository holds the models and solutions developed by Pablo Barros based on emotion recognition and learning.

The KEF framework was developed to facilitate the planing and fast prototyping of different scientific experiments. All the examples in this repository use the KEF framework.



**Pre-requisites**

tensorflow, keras, matplotlib, h5py, opencv-python, librosa, pillow, imgaug, python_speech_features, hyperas

If you want to run on a GPU, install tensorflow-gpu instead of tensorflow

**Instructions**

Each of the examples here run within the KEF framework. Also, each example needs a specific dataset which is not available here. All the demos and examples here run on Python 2.7.


**Hand gesture recognition**

- NCD_VisionNetwork_SobelXY.py: Multichannel Convolution Neural Network for hand posture recognition using the NCD dataset (Barros et al., 2014)

**Auditory emotion recognition**

- OMG_Emotion_Audio_MelSpectrum.py: Audio Channel for the OMG-Emotion dataset (Barros et al., 2018)
- RAVDESS_Audio_MelSpectrum_Channel.py: Audio Channel for the RAVDESS dataset (Barros et al., 2018)

**Visual emotion recognition**

- OMG_Emotion_Face.py: Face Channel for the OMG-Emotion dataset (Barros et al., 2018)
- FERPlus_Vision_FaceChannel.py: Face Channel for the FERPlus dataset (Barros et al., 2018)

**Crossmodal emotion recognition**

- OMG_Emotion_Crossmodal.py: Cross Channel for the OMG-Emotion dataset (Barros et al., 2018)
- RAVDESS_CrossNetwork_MelSpectrum_Channel.py: Cross Channel for the RAVDESS dataset (Barros et al., 2018)

**Personalized Affective Memory**
 - [Personalized Affective Memory Repository] - https://github.com/pablovin/P-AffMem
 -



**Trained Models**

 Each of the examples has a pre-trained model associated with it. Please refer to the TrainedModels folder.



**Ready-to-Run Demos**

 - [Visual Emotion Recognition](https://github.com/knowledgetechnologyuhh/EmotionRecognitionBarros/tree/master/Demos/VisualEmotionRecognition)


**Datasets**

Follows the links for different corpora that I developed or was involved on the development. Most of the examples here make use of these corpora:

- [OMG-Empathy Prediction](https://www2.informatik.uni-hamburg.de/wtm/omgchallenges/omg_empathy.html)
- [OMG-Emotion Recognition](https://www2.informatik.uni-hamburg.de/wtm/omgchallenges/omg_emotion.html)
- [Gesture Commands for Robot InTeraction (GRIT)](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html)
- [NAO Camera hand posture Database (NCD)](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html)

**Important references**
- Barros, P., Parisi, G., & Wermter, S. (2019, May). [A Personalized Affective Memory Model for Improving Emotion Recognition] (http://proceedings.mlr.press/v97/barros19a.html). In International Conference on Machine Learning (pp. 485-494). 
 - Barros, P., Barakova, E., & Wermter, S. (2018). [A Deep Neural Model Of Emotion Appraisal](https://arxiv.org/abs/1808.00252). arXiv preprint arXiv:1808.00252.
 - Barros, P., & Wermter, S. (2016). [Developing crossmodal expression recognition based on a deep neural model](http://journals.sagepub.com/doi/abs/10.1177/1059712316664017). Adaptive behavior, 24(5), 373-396. http://journals.sagepub.com/doi/full/10.1177/1059712316664017
 - Barros, P., & Wermter, S. (2017, May). [A self-organizing model for affective memory. In Neural Networks (IJCNN)](https://www2.informatik.uni-hamburg.de/wtm/publications/2017/BW17/Barros-Affective_Memory_2017-Webpage.pdf), 2017 International Joint Conference on (pp. 31-38). IEEE.
 - Barros, P., Jirak, D., Weber, C., & Wermter, S. (2015). [Multimodal emotional state recognition using sequence-dependent deep hierarchical features](https://www.sciencedirect.com/science/article/pii/S0893608015001847). Neural Networks, 72, 140-151.
 - Barros, P., Magg, S., Weber, C., & Wermter, S. (2014, September). [A multichannel convolutional neural network for hand posture recognition](https://www2.informatik.uni-hamburg.de/wtm/ps/Barros_ICANN2014_CR.pdf). In International Conference on Artificial Neural Networks (pp. 403-410). Springer, Cham.
 - [All the references](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)



**License**

All the examples in this repository are distributed under the Creative Commons CC BY-NC-SA 3.0 DE license. If you use this corpus, you have to agree with the following itens:

- To cite our associated references in any of your publication that make any use of these examples.
- To use the corpus for research purpose only.
- To not provide the corpus to any second parties.

**Contact**

Pablo Barros - barros@informatik.uni-hamburg.de

- [http://pablobarros.net](http://pablobarros.net)
- [Uni Hamburg Webpage](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/barros.html)
- [Google Scholar](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)
