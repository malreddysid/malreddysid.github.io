---
title:  "Mobile Friendly Deep Learning"
date:   2017-20-02 10:18:00
description = "A study on techniques to improve the run time of deep networks on mobiles"
---

In the past few years, convolutional neural networks have risen to popularity due to their effectiveness in various computer vision tasks. Networks like AlexNet, VGGNet, and GoogleNet have been proved to be very effective for image classification. Using a similar approach, networks like FCN, and SegNet have been shown to be effective in semantic segmentation whereas networks like FlowNet have been used to generate optical flow. Many experiments are also being done using CNNs for problems such as color correction, super resolution, and illuminant estimation.

Because of the huge accuracy boost when using CNNs, companies like Google and Facebook have already included them in their services. They do this by uploading our data to their servers, running the neural networks and then transferring the outputs. Even though this works for now, we still need to maintain an internet connection which may not always be available.
Then why not just run the networks on our devices? Because deep networks are very computationally intensive. They require multiple high-end GPUs in order to function in a real world scenario. Running them on a CPU takes multiple seconds, even longer on a mobile phone. Also, their models are huge in size. Loading a large model causes delay and consumes high amount of power as it would have to be stored in the DRAM. Herein lies the Achilles’ heel of CNNs.

But all hope is not lost. Many researchers are focusing on this very problem.

There are two ways to make deep CNNs faster.

1. Make them shallow.
2. Improve the underlying implementation of the convolutional layers.

## Shallow Networks

The first question that comes to mind when thinking about making deep nets shallow is why do we need to make them deep in the first place?
The answer is that any network needs to learn a complex function which maps the inputs to the output labels, and the present learning methodologies are not capable of teaching a shallow network the complex function directly. Prof. Rich Caruana discusses this in his paper [“Do Deep Nets really need to be Deep?”](http://arxiv.org/abs/1312.6184). He proves that by training a given dataset using an ensemble of deep nets and then transferring the function learned on these deep nets into a shallow net, it is possible to match the accuracy of the shallow net to that of the ensemble. However, simply training the shallow net using the original data gives an accuracy which is significantly lower. Thus, by transferring the function we can make the shallow network learn a much more complex function which otherwise would not have been possible by training directly on the dataset.

But how do we transfer the learned function? We run the dataset through the ensemble and store the logits (the values that are input to the SoftMax layer). Then we train the shallow network using the logits instead of the original labels. This can be intuitively understood because by training on the logits, we are not only training the model on the right answer but also on the “almost right” answers as well as “not at all right” answer.

  *For example given an image of a car to the ensemble, the logit value of it being a truck would much higher than it being a flower. Lets assume that the logits are car — 3, truck — 0.1, flower — 0.00001. If we train the shallow network using these probabilities, we are telling it that this image has a signifiant probability of being a car or a truck but it is definitely not a flower. So while training we are presenting the model with much more information than we would have if we had used labels.*

Dr. Geoffrey Hilton took it a step further and proposed that we should use the soft targets (weighted SoftMax function outputs) instead of logits. In his paper [“Distilling the Knowledge in a Neural Network”](https://arxiv.org/abs/1503.02531) he proposes that this provides more information to be learned by the shallow model.

## Faster Networks and Smaller Models
The amount of research being done on improving the run time of CNNs and compressing their models is staggering. I’m going to cover two major techniques being used for the same in this article.

* **Hardware Accelerators:**

Because CNNs run on images and each kernel in a layer is independent of the others, they are parallelizable and are the perfect candidates for hardware acceleration. Since each kernel in a layer is independent of the other kernels, they can be run on different cores or even different machines. Having dedicated hardware means power and speed improvement at the cost of area and price. But the effectiveness of deep learning is motivating hardware manufacturers to invest in developing such hardware. [Qualcomm zeroth](https://www.qualcomm.com/invention/cognitive-technologies/zeroth) and [Nvidia Digits](https://developer.nvidia.com/digits) are examples of such accelerators.

* **Mathematical Techniques:**

Since a convolutional neural network is basically a series of tensor operations, we can use tensor rank decomposition techniques to decrease the number of operations that need to be done for each layer. The paper [“Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications”](http://arxiv.org/abs/1511.06530) uses Variational Bayesian Matrix Factorization for rank selection and tucker-2-decomposition to split each layer into three layers.

  *For a convolutional layer of size T x S x K x K, rank selection is done for the 3rd and 4th axis (P and Q). Then this layer is decomposed into three different layers (P x S x 1 x 1, Q x P x K x K, T x Q x 1 x 1).*

This type of architecture is also found in ResNet, SqueezeNet and the inception layers in GoogleNet. It can be intuitively justified by considering that the input layers are correlated. So their redundancy can be removed by properly combining them with 1 x 1 layers. After the core convolution, they can be expanded for the next layer. The loss in accuracy due to this operation is compensated using fine-tuning.

Alternatively, techniques like pruning and weight sharing are used to compress the model thereby decreasing the model size as detailed in [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149). The author claims that the network learns redundant connections during training. So he proposes to remove such connections and keep only the most informative ones. He does this by removing connections with weights below a certain threshold and finetuning the remaining weights. Using this technique he achieves a 9x reduction in parameters for AlexNet. He also uses k-means clustering to identify weights that can be shared in a single layer.

With this, I conclude my post on making deep learning “mobile friendly”. Please let me know if I you have any suggestions.

**P.S.** Check out Prof. Caruana’s talks [here](http://research.microsoft.com/apps/video/default.aspx?id=103668) and [here](http://research.microsoft.com/apps/video/default.aspx?id=232373&r=1).

**P.P.S.** Dr. Hilton also shows that the model can learn to recognise inputs that it has never seen before just by inferring its structure from the soft targets. He calls it Dark Knowledge. Check out his [talk](https://www.youtube.com/watch?v=EK61htlw8hY) if you’re interested.

**Update:** Prof. Caruana’s published a new paper — [“Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?”](https://arxiv.org/abs/1603.05691) in which he concludes that shallow nets can emulate deep nets given that they have multiple convolutional layers.