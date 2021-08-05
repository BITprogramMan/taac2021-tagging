# 0.简介
多模态视频标签模型框架

# 1. 代码结构
- configs--------------------# 模型选择，参数文件
- src------------------------# 数据加载/模型相关代码
- scripts--------------------# 数据预处理/训练/测试脚本
- checkpoints----------------# 模型权重/日志
- pretrain_models-----------------# 预训练模型
- utils----------------------# 工具脚本
- train.py----------------------# 训练代码
- inference.py----------------------# 测试代码
- init.sh----------------------# 环境初始化的脚本
- train.sh----------------------# 启动训练的脚本
- inference.sh----------------------# 启动测试的脚本
- ReadMe.md

**补充说明：**dataset位于VideoStructuring文件夹下，与baseline相同，不同的是重新提取了视频特征，替换原来的特征

# 2. 环境配置

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./init.sh && ./init.sh run
```

# 3. 训练流程
## 3.1 加载预训练模型参数

在视频中提取到的文本使用robert模型，替换了baseline中的bert模型。所有需要的预训练模型都在pretrain_models文件夹下,预训练的robert模型以及用于视频特征提取的ViT模型可用百度网盘下载[（密码：t8lm）](https://pan.baidu.com/s/1cdydO1AF7zD8gCO2qr2jmA)

## 3.2 数据预处理
本实验只使用视频、语音、文本三种模态信息（不包含image），对于语音与文本特征，使用baseline提供的特征，本实验重新提取了视频特征，使用了vision Transformer（ViT）提取**测试集**视频特征运行以下代码：

```shell
python feat_extract_main.py --test_files_dir ../dataset/videos/test_5k_2nd
## 提取训练集特征改变test_files_dir参数
```

## 3.3 启动训练
本实验使用了8个模型做bagging集成，每个模型使用了5折交叉验证的方法。训练步骤以及运行方式如下：

首先训练一个模型产生soft label用于后续模型训练的正则化

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step1.sh && ./train_step1.sh train
```

运行上述代码将会在；主目录下产生`train_soft_label.csv`

然后按顺序训练8个5折交叉验证模型

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step2.sh && ./train_step2.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step3.sh && ./train_step3.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step4.sh && ./train_step4.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step5.sh && ./train_step5.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step6.sh && ./train_step6.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step7.sh && ./train_step7.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step8.sh && ./train_step8.sh train
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./train_step9.sh && ./train_step9.sh train
```

然后分别做测试：

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step2.sh && ./inference_step2.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step3.sh && ./inference_step3.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step4.sh && ./inference_step4.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step5.sh && ./inference_step5.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step6.sh && ./inference_step6.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step7.sh && ./inference_step7.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step8.sh && ./inference_step8.sh test
```

```shell
cd ~/notebook/VideoStructuring/MultiModal-Tagging && sudo chmod a+x ./inference_step9.sh && ./inference_step9.sh test
```

最后将八个模型的结果平均

```shell
python merge_result.py --result1 'step2/bagging_step2.json' \
                       --result2 'step3/bagging_step3.json' \
                       --result3 'step4/bagging_step4.json' \
                       --result4 'step5/bagging_step5.json' \
                       --result5 'step6/bagging_step6.json' \
                       --result6 'step7/bagging_step7.json' \
                       --result7 'step8/bagging_step8.json' \
                       --result8 'step9/bagging_step9.json' \
                       --output_json 'final_result.json' 
```

### 说明

+ 主目录为`~/notebook/VideoStructuring/MultiModal-Tagging`，dataset文件夹位于`~/notebook/VideoStructuring/`,结构如下：

--dataset

---------tagging

---------------GroundTruth

---------------tagging_dataset_train_5k

---------------tagging_dataset_test_5k_2nd

---------videos

---------------train_5k

---------------test_5k_2nd

---------label_id.txt

### 时间

+ 测试集特征提取时间为320分钟

+ 测试集预测推理时间为140分钟

  
  
  