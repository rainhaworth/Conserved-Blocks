# transformer.py modified to use sparse attention from bigbird/core/attention.py
import random, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.keras.backend as K

# import modified BigBird attention
import tfv2transformer.attention as attention

try:
	from tqdm import tqdm
	from dataloader import TokenList, pad_to_longest
	# for transformer
except: pass

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
	def __init__(self, attn_dropout=0.1):
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):   # mask_k or mask_qk
		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/x[2])([q, k, temper])  # shape=(batch, q, k)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, dropout, mode=0):
		self.mode = mode
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention()
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		return outputs, attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, length=1024, block_size=64, dropout=0.1):
		# sparse attention
		# using d_model for size_per_head
			# update: using d_model // n_head
		# need to set from_seq_length and to_seq_length
		# and i think they need to be multiples of from_block_length and to_block_length
		# from_ and to_ should match
		attn_type = 'block_sparse'
		num_rand_blocks = 3
		self.length = length
		self.block_size = block_size
		self.self_att_layer = attention.MultiHeadedAttentionLayer(
			attn_type, n_head, d_model // n_head, num_rand_blocks,
			self.length, self.length, self.block_size, self.block_size,
			dropout)

		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, mask=None):
		# band-aid: don't return attention, set to none
		slf_attn = None
		# TODO: add training param, length + block size params
		training = True
		# mask order: attention (unused), band, from, to, from_blocked, to_blocked
		block_mask = tf.reshape(mask, (-1, self.length//self.block_size, self.block_size))
		from_mask = tf.reshape(mask, (-1, 1, self.length, 1))
		to_mask = tf.reshape(mask, (-1, 1, 1, self.length))
		band_mask = attention.create_band_mask_from_inputs(block_mask, block_mask)
		# call attention layer
		output = self.self_att_layer(enc_input, enc_input, [
			None, band_mask, from_mask, to_mask, block_mask, block_mask
			], training=training)
		# reshape
		out_shape = tf.concat([tf.shape(output)[:-2], [enc_input.shape[-1]]], axis=0)
		output = tf.reshape(output, out_shape)
		# final layers
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, length=1024, block_size=64, dropout=0.1):
		# sparse attention
		# might need to set use_bias and initializer_range
		attn_type = 'block_sparse'
		num_rand_blocks = 3
		self.length = length
		self.block_size = block_size
		self.self_att_layer = attention.MultiHeadedAttentionLayer(
			attn_type, n_head, d_model // n_head, num_rand_blocks,
			self.length, self.length, self.block_size, self.block_size,
			dropout)
		self.enc_att_layer = attention.MultiHeadedAttentionLayer(
			attn_type, n_head, d_model // n_head, num_rand_blocks,
			self.length, self.length, self.block_size, self.block_size,
			dropout)
		
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer1 = LayerNormalization()
		self.norm_layer2 = LayerNormalization()
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
		if dec_last_state is None: dec_last_state = dec_input
		# band-aid: set attention return values to none
		slf_attn = None
		enc_attn = None
		# TODO: add training param
		training=True
		# make masks; hopefully you can just copy encoder for both
		self_block_mask = tf.reshape(self_mask, (-1, self.length//self.block_size, self.block_size))
		self_from_mask = tf.reshape(self_mask, (-1, 1, self.length, 1))
		self_to_mask = tf.reshape(self_mask, (-1, 1, 1, self.length))
		self_band_mask = attention.create_band_mask_from_inputs(self_block_mask, self_block_mask)

		enc_block_mask = tf.reshape(enc_mask, (-1, self.length//self.block_size, self.block_size))
		enc_from_mask = tf.reshape(enc_mask, (-1, 1, self.length, 1))
		enc_to_mask = tf.reshape(enc_mask, (-1, 1, 1, self.length))
		enc_band_mask = attention.create_band_mask_from_inputs(enc_block_mask, enc_block_mask)

		# self attention
		output = self.self_att_layer(dec_input, dec_last_state, [
			None, self_band_mask, self_from_mask, self_to_mask, self_block_mask, self_block_mask 
		], training=training)
		# reshape and norm
		out_shape = tf.concat([tf.shape(output)[:-2], [dec_input.shape[-1]]], axis=0)
		output = tf.reshape(output, out_shape)
		x = self.norm_layer1(Add()([dec_input, output]))
		# encoder attention
		output = self.enc_att_layer(x, enc_output, [
			None, enc_band_mask, enc_from_mask, enc_to_mask, enc_block_mask, enc_block_mask
		], training=training)
		# reshape and norm
		out_shape = tf.concat([tf.shape(output)[:-2], [x.shape[-1]]], axis=0)
		output = tf.reshape(output, out_shape)
		x = self.norm_layer2(Add()([x, output]))
		# feedforward
		output = self.pos_ffn_layer(x)
		return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask

class SelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, length=1024, block_size=64, dropout=0.1):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, length, block_size, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
		if return_att: atts = []
		mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(src_seq)
		x = src_emb
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x

class Decoder():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, length=1024, block_size=64, dropout=0.1):
		self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, length, block_size, dropout) for _ in range(layers)]
	def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
		x = tgt_emb
		# temporarily use 1D masks
		#self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
		#self_sub_mask = Lambda(GetSubMask)(tgt_seq)
		#self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		#enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
		# i guess these are the same here lol
		self_mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(tgt_seq)
		enc_mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(src_seq)
		if return_att: self_atts, enc_atts = [], []
		for dec_layer in self.layers[:active_layers]:
			x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
			if return_att: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		return (x, self_atts, enc_atts) if return_att else x

class ContrastiveEncoder:
	def __init__(self, tokens, len_limit, d_model=256, \
			  d_inner_hid=512, n_head=4, layers=2, \
			  length=1024, block_size=64, dropout=0.1):
		self.tokens = tokens
		self.len_limit = len_limit
		self.d_model = d_model
		self.layers = layers
		self.length = length
		d_emb = d_model
		
		self.src_loc_info = True

		d_k = d_v = d_model // n_head
		assert d_k * n_head == d_model and d_v == d_k

		self.pos_emb = PosEncodingLayer(len_limit, d_emb) if self.src_loc_info else None

		self.emb_dropout = Dropout(dropout)

		self.word_emb = Embedding(tokens.num(), d_emb)

		self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, length, block_size, dropout)
		#self.target_layer = TimeDistributed(Dense(tokens.num(), use_bias=False))
		# TODO: add projection head
		# temporary hack: comment out target_layer and do this instead
		self.target_layer = Dense(tokens.num(), use_bias=False)
	
	def compile(self, optimizer='adam', active_layers=999, batch_size=8):
		src_seq_input = Input(shape=(None,), batch_size=batch_size, dtype='int32')
		#tgt_seq_input = Input(shape=(None,), dtype='int32')

		src_seq = src_seq_input
		#tgt_seq = tgt_true = tgt_seq_input

		src_emb = self.word_emb(src_seq)
		#tgt_emb = self.o_word_emb(tgt_seq)

		if self.pos_emb: 
			src_emb = add_layer([src_emb, self.pos_emb(src_seq)])
			#tgt_emb = add_layer([tgt_emb, self.pos_emb(tgt_seq)])
		src_emb = self.emb_dropout(src_emb)

		enc_output = self.encoder(src_emb, src_seq, active_layers=active_layers)
		#dec_output = self.decoder(tgt_emb, tgt_seq, src_seq, enc_output, active_layers=active_layers)	
		final_output = self.target_layer(enc_output)
		# TODO: at least flatten this thing, ideally reduce the dimensionality

		def get_loss(y_pred, y_true):
			y_true = tf.cast(y_true, 'int32')
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss
		
		# bad for loop implementation of original multi-class n-pair loss paper
		# probably fine for reasonable batch size
		# untested, currently just shouldn't work bc i need to make design decision
		def contrastive_loss(out):
			loss = out
			N = len(loss)
			for i in range(N):
				sum = 0
				for j in range(N):
					if j != i:
						# pretend loss[j+1] and loss[i+1] give me the positive sample and j and i
						sum += tf.math.exp(tf.transpose(loss[i]) @ loss[j+1] - tf.transpose(loss[i]) @ loss[[i+1]])
				loss[i] = tf.math.log(1 + sum)
			loss = tf.reduce_sum(loss) / N
			return loss
		
		# ok i'm just using theirs now
		# this is modified from https://github.com/google-research/simclr/blob/master/tf2/objective.py
		# mainly removed TPU stuff bc i don't have convenient access to one
		# NOTE: (i'm pretty sure) this assumes `hidden` contains a stack of samples and their positive samples
		# i.e. class(hidden[0]) = class(hidden[batch_size]), class(hidden[1]) = class(hidden[batch_size + 1]), etc.
		def contrastive_batch_loss(hidden, hidden_norm=True, temperature=1.0):
			"""Compute loss for model.

			Args:
				hidden: hidden vector (`Tensor`) of shape (bsz, dim).
				hidden_norm: whether or not to use normalization on the hidden vector.
				temperature: a `floating` number for temperature scaling.
				strategy: context information for tpu.

			Returns:
				A loss scalar.
				The logits for contrastive prediction task.
				The labels for contrastive prediction task.
			"""
			LARGE_NUM = 1e9
			# Get (normalized) hidden1 and hidden2.
			if hidden_norm:
				hidden = tf.math.l2_normalize(hidden, -1)
			hidden1, hidden2 = tf.split(hidden, 2, 0)
			batch_size = tf.shape(hidden1)[0] #overwriting batch_size is fine probably

			# Create local labels. (removed TPU stuff and this is what's left)
			hidden1_large = hidden1
			hidden2_large = hidden2
			labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
			masks = tf.one_hot(tf.range(batch_size), batch_size)

			logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
			logits_aa = logits_aa - masks * LARGE_NUM
			logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
			logits_bb = logits_bb - masks * LARGE_NUM
			logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
			logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

			loss_a = tf.nn.softmax_cross_entropy_with_logits(
				labels, tf.concat([logits_ab, logits_aa], 1))
			loss_b = tf.nn.softmax_cross_entropy_with_logits(
				labels, tf.concat([logits_ba, logits_bb], 1))
			loss = tf.reduce_mean(loss_a + loss_b)

			return loss, logits_ab, labels

		def get_accu(y_pred, y_true):
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
			return K.mean(corr)
				
		#loss = get_loss(final_output, tgt_true)
		loss, logits, labels = contrastive_batch_loss(final_output)
		self.ppl = K.exp(loss)
		self.accu = get_accu(logits, labels) # not sure if this will work, comment out if it doesn't

		#self.model = Model([src_seq_input, tgt_seq_input], final_output)
		self.model = Model(inputs=src_seq_input, outputs=final_output)
		self.model.add_loss([loss])
		self.model.add_metric(self.ppl, name='ppl')
		#self.model.add_metric(self.accu, name='accu')

		self.model.compile(optimizer, None)

class PosEncodingLayer:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
						   weights=[GetPosEncodingMatrix(max_len, d_emb)])
	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])

if __name__ == '__main__':
	print('done')
