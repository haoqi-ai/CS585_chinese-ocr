"""
Generate data_path - label pairs. 
"""
import os
import sys
from tqdm import tqdm 

with open('./alphabet.txt') as fd:
    cvt_lines = fd.readlines()

cvt_dict = {}
for i, line in enumerate(cvt_lines):
    key = i
    value = line.strip()
    cvt_dict[key] = value

# python build.py ./train.txt > train_list.txt
# python build.py ./test.txt > test_list.txt
if __name__ == "__main__":
    fpath = sys.argv[1]
    with open(fpath) as fd:
        lines = fd.readlines()

    # print(len(cvt_dict), len(lines))
    for line in tqdm(lines):
        line_split = line.strip().split()
        img_path = line_split[0]
        label = ''
        for i in line_split[1:]:
            label += cvt_dict[int(i)-1]
        print(img_path, ' ', label)
