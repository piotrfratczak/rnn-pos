# rnn-pos
Experiments with different RNN architectures based on LSTM - basic LSTM / LSTM with dynamic graph based on POS tags / LSTM with input where POS tags are concatenated to word embeddings.


## Setup environment
1. Create conda environment:
```
conda env create -f environment.yml
```
2. Install GloVe dataset from: https://nlp.stanford.edu/data/glove.6B.zip, extract it and place it into data/glove/ directory.

## Configure run
Modify config files in config/ directory
...or use the default configuration by leaving it alone.  
**Notice:** When *run* parameter is set to "y" (yes), the models from the config file will be trained.

## Execute
```
python3 main.py
```

## Side note
Should work on Linux and MacOS.  
Windows is not recomended.
