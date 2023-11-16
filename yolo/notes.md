# 1. 问题
## 1.1 one-stage 与 two-stage的区别和联系是什么？如何定义one-stage与two-stage?

## 1.2 MAP指标，单看精度和recall不行吗？
Recall: 原始图像中，是不是每一个都检测到了。精度：每个框是否和真实值极度吻合。

## 

# 2. Stage.  

下图是one-stage的算法，可理解为一个神经网络输出一个回归（分类）结果。
<p align="center">
  <img src="./pics/onestage.jpg" alt="Image Description" width="60%"/>
</p>
下图是two-stage的算法，可以理解为在输出回归（分类）结果前，先进行一波预筛选。
<p align="center">
  <img src="./pics/twostage.png" alt="Image Description" width="60%"/>
</p>   

## 2.1 区别和优缺点
one-stage:   
1. 速度非常快
2. 通常情况效果不太好  

FPS: frame per second.  

# 3. Map
Map: 检测任务中经常提到的指标。Map越大，检测效果越好。综合衡量检测效果。  
## 3.1 IOU
Intersection of unit.
<p align="center">
  <img src="./pics/IOU.png" alt="Image Description" width="30%"/>
</p>   

$$
Precision = \frac{TP}{TP+FP}
$$
$$
Recall = \frac{TP}{TP+FN}
$$

## 3.2 Map
Map综合考虑P-R参数，得到的结果
<p align="center">
  <img src="./pics/map.png" alt="Image Description" width="60%"/>
</p> 

# 4. YOLO
## 4.1 YOLO-v1
yolov1中只有2中候选框，并从两个候选框中选择一个，然后进行微调。在预测时，除了框的几何参数，还有一个置信度，最后要将置信度小的框过滤掉。
<p align="center">
  <img src="./pics/yolov1_box.png" alt="Image Description" width="60%"/>
</p> 

### 4.1.1 网络
yolov1的网络架构如下：
<p align="center">
  <img src="./pics/yolov1_net.png" alt="Image Description" width="100%"/>
</p>   

特征图全连接层转化为4096*1，然后再连接一个1470*1的全连接网，再reshape成7*7*30，生成7*7的格子，每个格子中预测出30个值。  
举例：生成一个框，Box1: x1, y1, w1, h1, c1，其中x1, y1不是图片中的坐标值，而是经过整个图像归一化后的0到1的值，表示相对整个图像中的位置，c1表示置信度。  
30个参数，那么Box1 + Box2有10个参数，剩下的20，表示在此数据集中，有20个分类。  
最终网格的大小 （S * S）*(B*5+C)
### 4.1.2 损失函数

<p align="center">
  <img src="./pics/yolov1_loss.png" alt="Image Description" width="100%"/>
</p>

$$
L = L_1 + L_2 + L_3
$$

#### 4.1.2.1 位置误差
位置误差损失函数如下：
$$
L_1 = \lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\amalg_{ij}^{obj}(x_i-\hat{x_i})^2 + (y_i - \hat{y_i})^2 + \lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\amalg_{ij}^{obj}(\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h_i}})^2
$$  
其中$S^2$在yolov1中表示最后网络输出的7*7*30中的 __7*7__ , B 表示2个框，$\amalg$ 表示在两个候选框中选择IOU计算后更贴切的那个框。 $\sqrt{w_i}-\sqrt{\hat{w_i}}$ 的计算，与坐标点，$x_i, y_i$ 不同的原因是，减小因为框的大小导致的最小二乘损失函数所引起的不好的影响。如 5 *5 的框误差值为1和 1 * 1 的框，误差值为1的严重程度应该是不一样的，当数值小的时候，变化敏感，数值较大时，变化不那么敏感。解决对小物体检测不敏感的问题。
#### 4.1.2.2 置信度误差
置信度误差如下：  
包含障碍物的损失函数：
$$
L_{21} = \sum_{i=0}^{S^2}\sum_{j=0}^{B}\amalg_{ij}^{obj}(C_i-\hat{C_i})^2
$$
背景信息的损失函数:
$$
L_{22} = \lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\amalg_{ij}^{obj}(C_i-\hat{C_i})^2
$$
$$
L_2 = L_{21} + L_{22}
$$
#### 4.1.2.3 分类损失函数
$$
L3 =  \sum_{i=0}^{S^2}\amalg_{ij}^{obj}\sum_{c\in{classes}}(p_i(c)-\hat{p_i}(c))^2
$$

### 4.1.3 非极大值抑制
多个候选框，选择置信度最大的框

### 4.1.4 问题
1. 无法检测重合在一起的物体，由其算法选择2个Box导致的？-> TODO
2. 没有考虑的小物体的检测，长宽比的选择单一

## 4.2 YOLO-V2
### 4.2.1 迭代更改点
1. 舍弃dropout，卷积后全部加入Batch Normalization
2. 网络的每一层的输入都做归一化，收敛更加容易  
3. 经过BN的网络会有2%的提升
4. 使用了更大的分辨率，map提升了4个百分点
5. 网络机构：DarkNet, 没有FC层，5次降采样，1*1卷积（全连接层容易过拟合，全连接层训练较慢）
6. Darknet实际输入是416*416（为了让416/32=奇数），所有卷积核比较小，使得感受野比较大， 1 * 1的卷积比较省参数
### 4.2.2 先验框
faster-rcnn有 3* 3=9种先验框，但是比例是1：1， 1：2， 1:3  
yolov2 __聚类提取先验框__ $d(box, centroids)=1-IOU(box, centroids)$, 生成的先验框和真实的框更接近，k=5.  
此处的距离公式，如果用欧式距离，大的框产生的差异情况比较大，小的框产生差异比较小，可能算出来的值与框的大小相关，这是我们不想看到的。 所以此处使用的距离为 1-IOU。  
<p align="center">
  <img src="./pics/聚类先验框.png" alt="Image Description" width="60%"/>
</p>

当k值继续增大后，对网络模型的性能提升是非常有限的，所以yovov2选择了5个先验框的超参数。  
通过引入Anchor box， 使得预测的box更多，但是对map值提升不大，但是对recall的提升较多。  
### 4.2.3 偏移量计算
yolov2没有直接使用偏移量计算，而是选择相对 grid cell 的偏移量。
<p align="center">
  <img src="./pics/偏移量计算.png" alt="Image Description" width="60%"/>
</p>

计算公式如下：
$$
b_x = \sigma(t_x) + c_x \\
b_y = \sigma(t_y) + c_y  \\
b_w = p_we^{t_w} \\
b_h = p_he^{t_h}
$$
其中 $\sigma()$ 函数将t_x, t_y控制在0到1的范围内，那么在进行偏移量计算时，锚框就不会偏移太多。对于b_w使用指数计算的原因是一开始我们对预测值计算使用对数操作。
> 此处假设偏移出去了，代码是如何过滤掉偏移出去的候选框的？
### 4.2.4 感受野
当最后一层的感受野太大的时候，小目标可能丢失。
<p align="center">
  <img src="./pics/感受野.png" alt="Image Description" width="30%"/>
</p>

使用3个 3* 3的卷积感受野是 7 *7，与1个 7 * 7的卷积核的区别，主要是参数会更少。（可参考VGG的设计思路） 
### 4.2.5 特征图融合 
在yolov2做了一个特征的融合，将最后一个特征图与倒数第二个特征图进行融合。
<p align="center">
  <img src="./pics/特征融合.png" alt="Image Description" width="60%"/>
</p>

### 4.2.6 多尺度
yolov2增加了多尺度的功能

## 4.3 YOLO-V3
### 4.3.1 改进
1. 更改网络结构，更适合小目标检测
2. 特征做的更加细致，融入多持续特征图信息预测不同规格物体
3. 先验框更丰富（v1中2个框，v2中5个，v3中9个）
4. Softmax改进，预测多标签任务
### 4.3.2 多scale

<p align="center">
  <img src="./pics/多scale.png" alt="Image Description" width="80%"/>
</p>   

特征图 52： 小目标  
特征图 26： 中目标  
特征图 13： 大目标  
设计3种不同的特征图，再选择3种不同的候选框  

<p align="center">
  <img src="./pics/V3特征图融合.png" alt="Image Description" width="60%"/>
</p> 

### 4.3.3 Darknet 53
<p align="center">
  <img src="./pics/Darknet53.png" alt="Image Description" width="100%"/>
</p> 

1. 没有池化层和全连接层
2. 下采样通过 stride 为2实现
3. 3种不同的scale，更多的先验框
13 * 13 *3 *（80 + 4 +1） -> 先验框长 * 先验框宽 * 3个scale box * (80个类别 + 每个框 x,y,w,h + 背景)

### 4.3.4 softmax改进
物体检测任务中可能一个物体有多个标签