# CS585 Team Project: Chinese OCR through a CRNN Architecture with Attention

**Established by Hao Qi, Ziye Chen, Xi Chen**

This project presents a Convolutional Recurrent Neural Network (CRNN) with an attention mechanism tailored for Chinese scene text recognition. It overcomes challenges such as variable illumination and text distortions. Enhanced by traditional computer graphics methods and deep learning, the model we built from scratch can precisely recognize single-line modern Chinese text, offering a foundation for multilingual text recognition assistant systems.

![workflow](https://github.com/haoqi-ai/CS585_chinese-ocr/blob/main/workflow.png)  

# Run the demo
You can run `main.py` directly, or using the command line.

Assuming your current working directory is "CS585_chinese-ocr"：  
```bash
python main.py
```

# Train a new model

* Download [Synthetic Chinese String Dataset](https://github.com/senlinuc/caffe_ocr) on [Baidu Netdisk](https://pan.baidu.com/s/1bHRP2eAcU8a7ff0n-VTX_A) with the share code ***2anx***.  

* Create `train_list.txt` and `test_list.txt` as the following format:
```
<path_to_the_pictures> <ground_truth_characters>
...
```
You can use the "[data/build.py](https://github.com/haoqi-ai/CS585_chinese-ocr/blob/main/data/build.py)" script to create these two lists or finish it yourself.
```
cd data
python build.py SyntheticChineseStringDataset/train.txt > train_list.txt
python build.py SyntheticChineseStringDataset/test.txt > test_list.txt
```

* Start training
```
python train.py --train_list train_list.txt --val_list test_list.txt
``` 

# Reference
[https://github.com/senlinuc/caffe_ocr](https://github.com/senlinuc/caffe_ocr)
[https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
[https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
