# Deep CNN for Estimation of Remaining Useful Life
Inspired by Babu, G. S., Zhao, P., &amp; Li, X. L. (2016, April). Deep convolutional neural network 
based regression approach for estimation of remaining useful life. In _International conference on database systems for 
advanced applications_ (pp. 214-228). Springer, Cham.  
_Author: Jiaxiang Cheng, Nanyang Technological University, Singapore_

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />

## Environment
```
python==3.8.8
pytorch==1.8.1
pandas==1.2.4
scikit-learn==0.23.2
numpy==1.20.1
matplotlib==3.3.4
scipy==1.6.2
```

## Usage
You may simply give the following command for both training and evaluation:
```
python main.py
```
Then you will get the following running information:
```
...

Epoch :  30      loss : 3.285    RMSE = 34.636   Score = 14473
Epoch :  31      loss : 3.277    RMSE = 34.599   Score = 14815
Epoch :  32      loss : 3.269    RMSE = 34.95    Score = 12690
Epoch :  33      loss : 3.259    RMSE = 32.885   Score = 6656
Epoch :  34      loss : 3.25     RMSE = 32.354   Score = 5344
Epoch :  35      loss : 3.241    RMSE = 32.318   Score = 4898

...
```
As the model and data sets are not heavy, the evaluation will be conducted after each
training epoch to catch up with the performance closely.
The prediction results will be saved in the folder ```_trials```.

## Citation
[![DOI](https://zenodo.org/badge/360762936.svg)](https://zenodo.org/badge/latestdoi/360762936)

## License
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
