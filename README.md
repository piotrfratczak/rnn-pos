# rnn-pos
Experiments with different RNN architectures based on LSTM - basic LSTM / LSTM with dynamic graph based on POS tags / LSTM with input where POS tags are concatenated to word embeddings.


## Setup environment
1. Create conda environment:
```
conda env create -f environment.yml
```
2. Install GloVe dataset from: https://nlp.stanford.edu/data/glove.6B.zip, extract it and place it into data/glove/ directory.

## Configure run
Modify json config files in configs/ directory
...or use the default configuration by leaving it alone.  
**Notice:** When *run* parameter is set to "y" (yes), the models from the config file will be trained.

## Execute
```
python3 main.py
```

## Side note
Should work on Linux and MacOS.  
Windows is not recomended.

## Notebook
Alternatively, you can run the project in a Google Colab notebook with [this script](notebook/rnn-pos.ipynb).
