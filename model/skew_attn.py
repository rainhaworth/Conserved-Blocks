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

class SkewedAttention():
	def __init__(self, length, attn_dropout=0.1):
		self.length = length
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k): # TODO: add mask support for len < maxlen
		# compute attention matrix as with scaled dot product attention
		# equivalent but slightly slower: attn = tf.matmul(q, k, transpose_b=True) / temper
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/x[2])([q, k, temper])  # shape=(batch, q, k)
		
		#if mask is not None:
		#	mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
		#	attn = Add()([attn, mmask])
		
		# skew matrix by extracting diagonals and adding to form a matrix of the same size
		attn = tf.linalg.diag_part(attn, k=(-(self.length-1), self.length-1))
		attn = attn[:,:self.length] + tf.concat([attn[:,self.length:], tf.zeros((tf.shape(attn)[0], 1, self.length))], axis=1)

		# compute scores by computing sums of skewed columns, i.e. diagonals
		scores = tf.reduce_sum(attn, axis=2)
		scores = tf.sort(scores)

		# return column sums and attention values
		return scores, attn

class SimpleSkewBinary:
	def __init__(self, tokens, d_model=256, length=1024, dropout=0.1):
		self.tokens = tokens
		self.d_model = d_model
		self.length = length
		d_emb = d_model

		#self.emb_dropout = Dropout(dropout)

		self.word_emb = Embedding(tokens.num(), d_emb)
		#self.avgpool = AveragePooling1D(16, 16) # test; TODO: add argument if good
		self.skewatt = SkewedAttention(length)
		self.pred_layer = Dense(1, activation='sigmoid')

	def compile(self, optimizer='adam', lam_FN=50):
		src_seq_input = Input(shape=(self.length,), dtype='int32')
		tgt_seq_input = Input(shape=(self.length,), dtype='int32')

		src_seq = src_seq_input
		tgt_seq = tgt_seq_input

		src_emb = self.word_emb(src_seq)
		tgt_emb = self.word_emb(tgt_seq)

		#src_emb = self.avgpool(src_emb)
		#tgt_emb = self.avgpool(tgt_emb)

		scores, _ = self.skewatt(src_emb, tgt_emb)
		pred = self.pred_layer(scores)

		def bloom_loss(y_true, y_pred):
			binary_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)

			P_mask = tf.equal(y_true, 1)
			PN_mask = tf.less(y_pred, 0.5)
			true_FN = y_true * tf.cast(P_mask, 'int32') * tf.cast(PN_mask, 'int32')
			pred_FN = y_pred * tf.cast(P_mask, 'float32') * tf.cast(PN_mask, 'float32')

			FN_loss = tf.keras.losses.binary_crossentropy(true_FN, pred_FN, from_logits=False)

			return binary_loss + lam_FN*FN_loss

		self.model = Model([src_seq_input, tgt_seq_input], pred)
		self.model.compile(optimizer,
					 #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
					 loss=bloom_loss,
					 metrics=[
						 tf.keras.metrics.BinaryAccuracy(),
						 tf.keras.metrics.Recall()
						 ])

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])

if __name__ == '__main__':
	print('done')
