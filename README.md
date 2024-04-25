# CS585_chinese-ocr
This project presents a Convolutional Recurrent Neural Network (CRNN) with an attention mechanism tailored for Chinese scene text recognition. It overcomes challenges such as variable illumination and text distortions. Enhanced by traditional computer graphics methods and deep learning, the model we built from scratch can precisely recognize single-line modern Chinese text, offering a foundation for multilingual text recognition assistant systems.

# Run demo
You can run the **main.py** directly, or on the command line.

Asume your current work directory is "CS585_chinese-ocr"：  
```bash
python3 main.py
```

# Train a new model

* Download [Synthetic Chinese String Dataset](https://pan.baidu.com/s/1bHRP2eAcU8a7ff0n-VTX_A) with share code ***2anx***.  

* Create **train_list.txt** and **test_list.txt** as follow format
```
path/50843500_2726670787.jpg 情笼罩在他们满是沧桑
path/57724421_3902051606.jpg 心态的松弛决定了比赛
path/52041437_3766953320.jpg 虾的鲜美自是不可待言
```
You can use the "[data/build.py](https://github.com/haoqi-ai/CS585_chinese-ocr/blob/main/data/build.py)" script to create the two lists or finish it by yourself.
```
cd data
python3 convert_text_list.py SyntheticChineseStringDataset/train.txt > train_list.txt
python3 convert_text_list.py SyntheticChineseStringDataset/test.txt > test_list.txt
```

* Start training
```
#cd crnn_seq2seq_ocr_pytorch
python3 --train_list train_list.txt --eval_list test_list.txt --model ./model/crnn/ 
``` 
