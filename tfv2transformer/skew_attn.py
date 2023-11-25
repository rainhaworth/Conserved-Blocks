# skewed attention mechanism for identifying sequences with shared regions
# modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

#import random, os, sys
#import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K

# import modified BigBird attention
#import tfv2transformer.attention as attention

try:
	from tqdm import tqdm
	from dataloader import TokenList, pad_to_longest
	# for transformer
except: pass

class SkewedAttention():
	def __init__(self, length, attn_dropout=0.1):
		self.length = length
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k): # TODO: add mask support for len < maxlen
		# compute attention matrix as with scaled dot product attention
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/x[2])([q, k, temper])  # shape=(batch, q, k)
		
		#if mask is not None:
		#	mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
		#	attn = Add()([attn, mmask])
		
		# skew along last dim; if three's a better way to do this, do that instead
		attn = tf.stack([tf.roll(attn[:,i], -i, 1) for i in range(self.length)], axis=1)

		# compute column sums and sort
		colsums = tf.reduce_sum(attn, axis=1)
		colsums = tf.sort(colsums)

		# return column sums and attention values
		return colsums, attn

class SimpleSkewBinary:
	def __init__(self, tokens, d_model=256, length=1024, dropout=0.1):
		self.tokens = tokens
		self.d_model = d_model
		self.length = length
		d_emb = d_model

		#self.emb_dropout = Dropout(dropout)

		self.word_emb = Embedding(tokens.num(), d_emb)
		self.skewatt = SkewedAttention(length)
		self.pred_layer = Dense(1, activation='sigmoid')

	def compile(self, optimizer='adam'):
		src_seq_input = Input(shape=(self.length,), dtype='int32')
		tgt_seq_input = Input(shape=(self.length,), dtype='int32')

		src_seq = src_seq_input
		tgt_seq = tgt_seq_input

		src_emb = self.word_emb(src_seq)
		tgt_emb = self.word_emb(tgt_seq)

		colsums, _ = self.skewatt(src_emb, tgt_emb)
		pred = self.pred_layer(colsums)

		self.model = Model([src_seq_input, tgt_seq_input], pred)
		self.model.compile(optimizer,
					 tf.keras.losses.BinaryCrossentropy(from_logits=False), 
					 tf.keras.metrics.BinaryAccuracy())

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])

if __name__ == '__main__':
	print('done')
