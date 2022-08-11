# BERT
from transformers import BertTokenizer
import tensorflow_datasets as tfds
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def input_for_bert_model(df, tokenizer, max_seq_length):
    columns_list = df.columns.tolist()
    input_ids = np.zeros((len(df), max_seq_length))
    input_attention_masks = np.zeros((len(df), max_seq_length))
    
    if 'is_sarcastic' in columns_list:
        train_labels = np.zeros((len(df), 1))
        for i, labels in enumerate(df['is_sarcastic']):
            train_labels[i,:] = labels
    
    for i, sequence in enumerate(df['headline']):
        tokens = tokenizer.encode_plus(
            sequence,
            max_length = max_seq_length, # max length of the text that can go to BERT
            truncation=True, padding='max_length',
            add_special_tokens = True, # add [CLS], [SEP]
            return_token_type_ids = False, 
            return_attention_mask = True, # add attention mask to not focus on pad tokens
            return_tensors = 'tf'
        )
        input_ids[i,:], input_attention_masks[i,:] = tokens['input_ids'], tokens['attention_mask']
    
    if 'is_sarcastic' in columns_list:
        return input_ids, input_attention_masks, train_labels
    else:
        return input_ids, input_attention_masks
    
train_ids, train_attention_masks, train_labels = input_for_bert_model(train_dataframe, bert_tokenizer, max_length)
test_ids, test_attention_masks, test_labels = input_for_bert_model(test_dataframe, bert_tokenizer, max_length)
train_inputs = {"input_ids":train_ids[:22895], "attention_mask":train_attention_masks[:22895]}
train_outputs = train_labels[:22895]
valid_inputs = {"input_ids":train_ids[22895:], "attention_mask":train_attention_masks[22895:]}
valid_outputs = train_labels[22895:]
test_inputs = {"input_ids":test_ids, "attention_mask":test_attention_masks}
test_outputs = test_labels[22895:]
#Bert model initialization
from transformers import BertTokenizer, TFBertModel
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
sequence_output = bert_model(input_ids, attention_mask=attention_mask)[0][:,0,:]
x = tf.keras.layers.Dropout(0.1)(sequence_output)
out = tf.keras.layers.Dense(1, activation='linear', name="outputs")(x)
model4 = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model4.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
history4 = model4.fit(train_inputs, train_outputs, epochs=10, batch_size=8, validation_data=(valid_inputs, valid_outputs))
