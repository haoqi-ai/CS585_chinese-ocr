# CS585_chinese-ocr

**Made by Hao Qi, Ziye Chen, Xi Chen**

This project presents a Convolutional Recurrent Neural Network (CRNN) with an attention mechanism tailored for Chinese scene text recognition. It overcomes challenges such as variable illumination and text distortions. Enhanced by traditional computer graphics methods and deep learning, the model we built from scratch can precisely recognize single-line modern Chinese text, offering a foundation for multilingual text recognition assistant systems.

![Frame.jpg](https://github.com/haoqi-ai/CS585_chinese-ocr/blob/main/README_image/Frame.png?)  

# Run the demo
You can run `main.py` directly, or using the command line.

Assuming your current work directory is "CS585_chinese-ocr"：  
```bash
python main.py
```

# Train a new model

* Download [Synthetic Chinese String Dataset](https://github.com/senlinuc/caffe_ocr) on [Baidu Netdisk](https://pan.baidu.com/s/1bHRP2eAcU8a7ff0n-VTX_A) with share code ***2anx***.  

* Create `train_list.txt` and `test_list.txt` as the following format:
```
<path_to_the_pictures> <ground_truth_characters>
...
```
You can use the "[data/build.py](https://github.com/haoqi-ai/CS585_chinese-ocr/blob/main/data/build.py)" script to create the two lists or finish it by yourself.
```
cd data
python build.py SyntheticChineseStringDataset/train.txt > train_list.txt
python build.py SyntheticChineseStringDataset/test.txt > test_list.txt
```

* Start training
```
#cd crnn_seq2seq_ocr_pytorch
python3 --train_list train_list.txt --eval_list test_list.txt --model ./model/crnn/ 
``` 
