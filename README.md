# Information-Bottleneck
Analysis of the application of the Information Bottleneck principle to Deep Neural Networks (DNN) based on Shwartz-Ziv, R. and Tishby, N., “Opening the Black Box of Deep Neural Networks via Information”.

# PACKAGES

    numpy-1.26.4
    torch-2.2.0
    torchvision-0.17.0
    matplotlib-3.8.2
    scipy-1.12.0

# TODO

- Reproduce results from IB paper
- Extend to MNIST: tanh, relu
- Check discussion on https://openreview.net/forum?id=ry_WPG-A-
- Better MI estimators due to Kraskov 2003, Kolchinsky 2017 and Goldfeld 2019
- Regularization

# References

[Example project](https://github.com/fournierlouis/synaptic_sampling_rbm/blob/master/Rapport_Projet_Neurosciences___Synaptic_Sampling.pdf)

- Shwartz-Ziv & Tishby, 2017
    - HTML version: https://ar5iv.labs.arxiv.org/html/1703.00810
    - Talk: https://www.youtube.com/watch?v=bLqJHjXihK8
    - News article: https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/#
    - Code: https://github.com/ravidziv/IDNNs
    - Horrible code, unsupported

- Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox, On the Information Bottleneck Theory of Deep Learning, ICLR 2018.
    - They criticize the IB principle
    - Code: https://github.com/artemyk/ibsgd
    - Nice code but doesn't work 
        - _feed_targets: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    - Might want to check the way they compute MI

- Nice implementation of Tishby's paper
    - https://github.com/shalomma/PytorchBottleneck
    - Pytorch
    - Works
    - Test with latest version of Pytorch

- Simple implementation of Tishby's paper
    - https://github.com/stevenliuyi/information-bottleneck
    - Haven't tested it yet

- List of IB papers
    - https://github.com/ZIYU-DEEP/Awesome-Information-Bottleneck

# Micro-Project

Test Shwartz-Ziv & Tishby, 2017 ideas on real dataset (MNIST):
- binary classification
- multi-class
- do different regularizations impact compression phase?

# Related work

- Noam Slonim, Agglomerative Information Bottleneck, 1999
    - Bottom up version of the IB iterative algorithm
- Ravid Shwartz-Ziv, Information Flow in Deep Neural Networks, 2022
    - PhD thesis, sumarized many IB works
- Elad Schneidman, Analysing Neural Codes using the Information Bottleneck Method, 2001
<br><br><br>
- Andrew M. Saxe, **On the Information Bottleneck Theory of Deep Learning**, 2018
    - Criticize the IB principle (check how they estimate MI) [GitHub](https://github.com/artemyk/ibsgd)
- Ziv Goldfeld, **Estimating Information Flow in Deep Neural Networks**, 2019
    - Accurate estimation of IB in DNN
    - Takes issue with Opening the Black Box of Deep Neural Networks via Information
<br><br><br>
- Alexander A. Alemi, Deep Variational Information Bottleneck, 2019
    - IB as an objective function [GitHub](https://github.com/alexalemi/vib_demo) [Implementation 1](https://github.com/udeepam/vib) [Implementation 2](https://github.com/1Konny/VIB-pytorch)
- Zoe Piran, The Dual Information Bottleneck, 2020
    - Addresses shortcomings of the IB framework, dualIB as an objective function [GitHub](https://github.com/ravidziv/dual_IB)