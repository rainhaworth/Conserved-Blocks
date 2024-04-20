# skewed attention mechanism for identifying sequences with shared regions
# with downsampling for memory footprint reduction
# modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K

class SkewedAttention():
	def __init__(self, length=1024):
		self.length = length
	def __call__(self, q, k, d=16):
		# downsampling k; input shape = (bsz, maxlen), output shape = (bsz, maxlen/d)
		k = k[:,::d]
		# reordering q to preserve diagonals
		q = tf.concat([q[:,i::d] for i in range(d)], 1)

		# compute attention matrix as with scaled dot product attention
		# equivalent but slightly slower: attn = tf.matmul(q, k, transpose_b=True) / temper
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/x[2])([q, k, temper])  # shape=(batch, q, k)
		
		# skew matrix by extracting diagonals and adding to form a matrix of the same size
		attn = tf.linalg.diag_part(attn, k=(-(self.length-1), self.length//d-1))

		# reassemble weird shaped downsampled matrix
		attn_partial = attn[:,:self.length//d] + tf.concat([attn[:,self.length:], tf.zeros((tf.shape(attn)[0], 1, self.length//d))], axis=1)
		attn = tf.concat([attn_partial, attn[:,self.length//d:self.length]], axis=1)

		# compute scores by computing sums of skewed columns, i.e. diagonals
		scores = tf.reduce_sum(attn, axis=2)
		scores = tf.sort(scores)

		# return column sums and attention values
		return scores, attn

class SimpleSkewBinary:
	def __init__(self, tokens, d_model=256, length=1024, ds=16):
		self.tokens = tokens
		self.d_model = d_model
		self.length = length
		self.ds = ds
		d_emb = d_model

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

		scores, _ = self.skewatt(src_emb, tgt_emb, self.ds)
		pred = self.pred_layer(scores)

		self.model = Model([src_seq_input, tgt_seq_input], pred)
		self.model.compile(optimizer,
					 tf.keras.losses.BinaryCrossentropy(from_logits=False), 
					 tf.keras.metrics.BinaryAccuracy())

if __name__ == '__main__':
	print('done')
