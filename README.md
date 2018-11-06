# DL4ETI
## Data
Dataset for our experiment will release soon.

## Requirements
1. Python version >= 3.6.5
2. Tensorflow >= 1.2.0
3. Keras >= 2.1.3  

## Usage
1. Use `python create_file_list.py` to create the __origial file list__.
2. Use `python four_random.py` or `python two_random.py` to create __random file list__ for four-class or binary classification.
3. Trian your models by using `python train_and_val.py`.
4. Caculate accuracy, sensitivity and specificity by using `python acc_sen_spe.py` when you have trained a model. 
5. Code in __Plot__ can help you plot figures below.
![Image text](https://github.com/ssea-lab/DL4ETI/blob/master/readme_img/CM.png)  
<div align='center'><img width="450" height="450" src="https://github.com/ssea-lab/DL4ETI/blob/master/readme_img/ROC.png"/></div>
