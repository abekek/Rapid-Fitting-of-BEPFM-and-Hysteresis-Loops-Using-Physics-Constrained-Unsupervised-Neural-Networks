# Rapid Fitting of Band-Excitation Piezoresponse Force Microscopy Using Physics Constrained Unsupervised Neural Networks

---

Alibek T. Kaliyev<sup>1,2</sup>, 
Ryan Forelli<sup>5</sup>, 
Pedro Sales<sup>7</sup>, 
Shuyu Qin<sup>1</sup>, 
Yichen Guo<sup>3</sup>, 
Olugbodi (FJ) Oluwafolajinmi<sup>5</sup>,
Andrew Zheng<sup>6</sup>,
Seda Ogrenci Memik<sup>12</sup>,
Michael W. Mahoney<sup>11</sup>, 
Amir Gholami<sup>11</sup>, 
Rama K. Vasudevan<sup>9</sup>, 
Stephen Jesse<sup>9</sup>, 
Nhan Tran<sup>10</sup>, 
Philip Harris<sup>8</sup>, 
Martin Takáč<sup>4</sup>, 
Joshua C. Agar<sup>3*</sup>

___

<sup>1</sup> Department of Computer Science and Engineering, Lehigh University, Bethlehem, PA 18015, USA

<sup>2</sup> College of Business, Lehigh University, Bethlehem, PA 18015, USA

<sup>3</sup> Department of Materials Science and Engineering, Lehigh University, Bethlehem, PA 18015, USA

<sup>4</sup> Department of Industrial and Systems Engineering, Lehigh University, Bethlehem, PA 18015, USA

<sup>5</sup> Department of Electrical and Computer Engineering, Lehigh University, Bethlehem, PA 18015, USA

<sup>6</sup> Department of Mechanical Engineering & Mechanics, Lehigh University, Bethlehem, PA 18015, USA

<sup>7</sup> Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology (MIT), Cambridge, MA, USA

<sup>8</sup> Department of Physics, Massachusetts Institute of Technology (MIT), Cambridge, MA, USA

<sup>9</sup> Center for Nanophase Materials Sciences, Oak Ridge National Laboratory, Oak Ridge, TN 37830, USA

<sup>10</sup> Fermi National Accelerator Laboratory, Batavia, IL 60510, USA

<sup>11</sup> University of California, Berkeley, Berkeley, CA 94720, USA

<sup>12</sup> Department of Electrical and Computer Engineering, Northwestern University, Evanston, IL 60208, USA

*joshua.agar@lehigh.edu

___

## Abstract

For nearly a decade, band-excitation piezoresponse force-based switching spectroscopy (BEPS) has been used to characterize ferroelectric switching and dynamic electromechanical responses of materials with nanoscale resolution. One of the key outputs of this technique is hyperspectral images of piezoelectric hysteresis loops, wherein there are one or more hysteresis loops at every pixel position. The challenge and dedication required to properly analyze data from these experiments have throttled the impact and widespread use of BEPS. To simplify the extraction of information from these datasets, a common approach involves fitting the piezoelectric hysteresis loops to an empirical function to parameterize the loops. This technique has several shortcomings:

It is computationally intensive, requiring more than 24 hours to process a single experiment on a single workstation with parallel processing.
It is highly dependent on prior estimates, which are difficult to compute to ensure optimization close to the global minimum.
It is unable to accommodate some of the complex features observed in piezoelectric hysteresis loops.

In an alternative approach, researchers have applied machine learning algorithms including principal component analysis, clustering algorithms, and non-negative matrix factorization to statistically address this problem. These algorithms are limited by their linear constraints, computational efficiency, and interpretability.

Our goal is to develop a fully unsupervised approach based on deep recurrent neural networks in the form of an autoencoder. This autoencoder will be able to learn a sparse and thus interpretable latent space of piezoelectric hysteresis loops, revealing detailed physical insight that will allude to results from other analysis techniques. This approach, however, is generally not applied due to the computational resources required for training. Here, we address this problem by developing generalized pre-trained models which can conduct feature extraction from piezoelectric hysteresis loops with minimal or potentially no training. We will achieve this by feeding a large database of noisy real piezoelectric hysteresis loops into the unsupervised model, with Residual Network architecture and Attention layers (widely used in Natural Language Processing), which will reproduce loops using the empirical function. After the training, we will extract the intermediate layer responsible for predicting parameters of loops. We will then benchmark the performance by validating the model on example open experimental datasets. We will determine how to best fine-tune these models using minimal computational resources to improve their efficacy on experimental data. By developing pre-trained models, we can significantly decrease the computational complexity of using these techniques.

Using this approach, it might be possible to deploy these methods for real-time analysis of BEPS, thus enabling experimentalists to improve their experimental efficiency and extract more information from these experiments. While our work focuses on developing models and benchmarking their efficacy in BEPS, this methodology could be adapted to other spectroscopic imaging techniques.
