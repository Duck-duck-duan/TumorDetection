<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200801105016867.png" alt="image-20200801105016867" style="zoom:70%;" />

Seed 具体使用？

augmentation: https://zhuanlan.zhihu.com/p/30197320

**[rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, brightness_range=[0.5, 1.5], horizontal_flip=True, vertical_flip=True]**

**ZCA whitening**：白化处理分PCA白化和ZCA白化，PCA白化保证数据各维度的方差为1，而ZCA白化保证数据各维度的方差相同。PCA白化可以用于降维也可以去相关性，而ZCA白化主要用于去相关性，且尽量使白化后的数据接近原始输入数据。

**width_shift_range**: Float, 1-D array-like or int。如果是float: fraction of total width, if < 1, or pixels if >= 1

**shear_range**: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)

<img src="https://pic3.zhimg.com/80/v2-0808ef3e12ab126663b8c6dabfa44a81_1440w.jpg" alt="img" style="zoom:50%;" />

**horizontal_flip**: Boolean. Randomly flip inputs horizontally.

**vertical_flip**: Boolean. Randomly flip inputs vertically.

**rescale**: rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations).

rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。

keras中**preprocess_input()**函数的作用是对样本执行 逐样本均值消减 的归一化，即在每个维度上减去样本的均值，对于维度顺序是channels_last的数据



**微调Fine-tuning**: https://zhuanlan.zhihu.com/p/35890660

Pre-trained model：预训练模型。现在我们常用的预训练模型就是他人用常用模型，比如VGG16/19，Resnet等模型，并用大型数据集来做训练集，比如Imagenet，COCO等<u>训练好的模型参数</u>。

正常情况下，常用的VGG16/19等网络已经是他人调试好的优秀网络，我们无需再修改其网络结构。

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200802090330482.png" alt="image-20200802090330482" style="zoom:60%;" />

CNN通常参数很多，几百万，在小数据集（小于参数数量）上训练CNN会极大地影响CNN泛化的能力，通常会导致过度拟合。

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200802094828586.png" alt="image-20200802094828586" style="zoom:50%;" />

因此，在实践中更经常的是，通过对我们拥有的较小数据集进行训练（即反向传播），对现有网络进行微调，这些网络是在像ImageNet这样的大型数据集上进行训练的，以达到快速训练模型的效果。假设我们的数据集与原始数据集（例如ImageNet）的上下文没有很大不同，预先训练的模型将已经学习了与我们自己的分类问题相关的特征。



<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200802142145509.png" alt="image-20200802142145509" style="zoom:67%;" />

或者softmax、relu



机器学习中，有很多优化方法来试图寻找模型的最优解。比较基础的有GD标准梯度下降法、SGD随机梯度下降法、BGD批量梯度下降法，还有Momentum、RMSProp、Adam算法、Adagrad算法等。

GD：遍历全部数据集算一次损失函数，求梯度，更新梯度。计算量开销大，计算速度慢。

SGD：每看一个数据就算一下损失函数，然后求梯度更新参数。速度快，但是收敛性能不太好，可能在最优店附近晃来晃去。

mini-batch GD：把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

Momentum：平滑处理。解决mini-batch GD的缺点，基于梯度的移动指数加权平均，v是动量，$\beta$是梯度累积的一个指数。该优化器的主要思想是对网络的参数进行平滑处理，让梯度的摆动度变得更小。dW和db分别是损失函数反向传播时候所求得的梯度，下面两个公式是网络权重向量和偏置向量的更新公式，$\alpha$是网络的学习率。当我们使用Momentum优化算法的时候，可以解决mini-batch SGD优化算法更新幅度摆动大的问题，同时可以使得网络的收敛速度更快。

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200802155621885.png" alt="image-20200802155621885" style="zoom:50%;" />

深度学习优化算法(Momentum, RMSProp, Adam)https://blog.csdn.net/willduan1/article/details/78070086

### Fine-tune InceptionV3 on a new set of classes

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(...)
```



Cross-entropy for machine learning

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200803104745609.png" alt="image-20200803104745609" style="zoom:50%;" />

The cross-entropy between two probability distributions, such as **Q from P**, can be stated formally as:

- H(P, Q)

where H() is the cross-entropy function, P may be the target distribution and Q is the approximation of the target distribution. Cross-entropy can be calculated using the probabilities of the events from P and Q, as follows:
$$
H(P, Q) = – \sum_{x \in X} P(x) * log_2(Q(x))
$$

#### 代码运行记录

##### 原本demo跑出来的：

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809052439668.png" alt="image-20200809052439668" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809052500704.png" alt="image-20200809052500704" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809052518414.png" alt="image-20200809052518414" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809052540571.png" alt="image-20200809052540571" style="zoom:50%;" />

##### 加了augmentation

都加：

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809062142682.png" alt="image-20200809062142682" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809062200073.png" alt="image-20200809062200073" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809062220520.png" alt="image-20200809062220520" style="zoom:50%;" />

修改参数：

1.去掉shear_range：

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809083758248.png" alt="image-20200809083758248" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809083821331.png" alt="image-20200809083821331" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200809083913153.png" alt="image-20200809083913153" style="zoom:50%;" />

去掉earlystopping：

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200810054028974.png" alt="image-20200810054028974" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200810054049604.png" alt="image-20200810054049604" style="zoom:50%;" />

<img src="/Users/duan/Library/Application Support/typora-user-images/image-20200810054008578.png" alt="image-20200810054008578" style="zoom:50%;" />

去了earlystopping太慢了，还加上earlystopping，去掉horizontal_flip 和 vertical_flip：

patience = 10看看会不会更快一点

准确度超级低，只有0.54

将patience = 20改回来

0.62

只有horizontal_flip



##### VGG + fine-tuning or other network + fine-tuning

不用Adam，用SGD，+weight decaying

Kaggle 以及 吴恩达 关于癌症检测的网络模型



**Question**

Applying data augmentation to medical images may sometimes make the images uninterpretable to human. For instance, a heart may not look like a heart after shearing the image. Would training the model with these uninterpretable images helpful to improving the model performance? Why do you think it is helpful/not helpful?

What are the advantages and disadvantages of finetuning a model for tumour detection?

What causes the gap between training accuracy/loss and test accuracy/loss? How do you reduce the gap between training accuracy/loss and test accuracy/loss?

