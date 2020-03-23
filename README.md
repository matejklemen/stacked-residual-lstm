# Overview
This repository contains the implementation of stacked residual LSTM seq2seq architecture 
with focus on paraphrase generation. The hope was to reproduce **Neural Paraphrase Generation with Stacked Residual LSTM Networks (Prakash et al., 2016)**, 
but I am not able to obtain the same results on MSCOCO as many things are ambiguously defined

Besides the main model type, the implementation can also be used for other various combinations of bidirectional LSTM, 
LSTM with attention, stacking LSTM and residual LSTM.


TODO:
- improve overall code style + usability of the project  
- add obtained results on COCO for a few models 

# Usage

Before doing anything, make sure to install the dependencies.  
`$ pip install -r requirements.txt`

## Data
The following instructions are for the MS Coco dataset, but they should be similar for any sequence-to-sequence data -
some additional changes might be required in the code.  

1. Download the 2014 train/val annotations from [COCO dataset website](http://cocodataset.org/#download). Create a 
folder `data/mscoco` in the project and put the two JSON files there.

2. Create splits and vocabulary. After running the command below, a `vocab.txt` file, `train_set`, `dev_set` and 
`test_set` folders will be created in `data/mscoco`.  In these folders, the space-tokenized source and target sequences 
will be written. The test folder will also contain reference streams for possible multi-reference evaluation.  
`$ python data_handler.py`  

**NOTE: The implementation assumes the use of following special tokens: `<PAD>`, `<BOS>`, `<EOS>` and `<UNK>`**. Keep 
this in mind if you are writing custom code using the implementation.


## Training a model
The training logic is inside the `train.py` file. **Currently, the code is still a bit of a WIP**, so you can't provide 
the model hyperparameters via command line options or a config file. To train a model, run:  
`$ python train.py`  

Following are some general options you can tweak:
- `DATA_DIR` = the directory of the dataset. This is expected to contain `train_set/train_src.txt`, 
`train_set/train_dst.txt`, `dev_set/dev_src.txt`, `dev_set/dev_dst.txt` (aligned sequences for train and dev set) 
and `vocab.txt` (vocabulary, token per line);  
- `MODEL_NAME` = descriptive name of your model. A folder with this name will be created in `DATA_DIR` and the best
model will be saved there;  
- `MAX_SEQ_LEN` = maximum length of generated sequence;  
- `BATCH_SIZE`, `NUM_EPOCHS` = self-explanatory;  
- `LOG_EVERY_N_BATCHES` = frequency of logging intermediate training loss (after every `LOG_EVERY_N_BATCHES` batches);  
- `TEACHER_FORCING_PROBA` = probability of using teacher forcing during training. **Important**: if set to `1.0`, 
teacher forcing will also be used evaluating on dev set.  

Following are some model-specific options you can tweak:
- `num_layers`: number of LSTM layers to use   
- `residual_layers`: sequence of layer ids after which the residue is added (0-based)  
- `inp_hid_size`: the size of input and hidden size in seq2seq encoder [**encoder-specific**];  
- `dropout`: dropout probability - applied after all but last layer  
- `bidirectional`: whether to use bidirectional encoder [**encoder-specific**]. **Important**: if set, the effective 
hidden size will be halved (i.e. each direction will get half dimensions)  
- `inp_size`, `hid_size`: input and hidden sizes of decoder. `hid_size` needs to be same as encoder's `inp_hid_size` 
[**decoder-specific**]  
- `num_attn_layers`: number of attention layers to use for decoding. Supported values are `0` (= no attention), `1` 
(= common attention for all decoder layers) or `num_layers` (= separate attention for each decoder layer) 
[**decoder-specific**]


## Testing (prediction)
The prediction logic is inside the `predict.py` file. The construction of models in this file should be done in the same 
 way as that in `train.py`. Also make sure the `MODEL_NAME` is set to the name of model which you trained using the 
 steps above. To generate sequences for the test set, run:  
`$ python predict.py`  


## Evaluation
To evaluate the generated sequences using SacreBLEU, use the `evaluate.py` script. Again, make sure `MODEL_NAME` is set 
to the name of the model you want to use.  
`$ python evaluate.py`

## Examples

#### Base LSTM
```python
enc_model = ResidualLSTMEncoder(vocab_size=...,
                                num_layers=1, residual_layers=None,
                                inp_hid_size=512,
                                dropout=...)
dec_model = ResidualLSTMDecoder(vocab_size=...,
                                num_layers=1, residual_layers=None,
                                inp_size=512,
                                hid_size=512,
                                dropout=...)
```

#### Residual LSTM
4-layered seq2seq LSTM with residue added after second layer.
```python
enc_model = ResidualLSTMEncoder(vocab_size=...,
                                num_layers=4, residual_layers=[1],
                                inp_hid_size=512,
                                dropout=...)
dec_model = ResidualLSTMDecoder(vocab_size=...,
                                num_layers=4, residual_layers=[1],
                                inp_size=512,
                                hid_size=512,
                                dropout=...)
```
