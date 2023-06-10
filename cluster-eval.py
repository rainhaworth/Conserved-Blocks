# TODO:
    # 1. Import saved model weights [done]
    # 2. Predict for arbitrary string with saved model; should produce the same string or very similar
        # 2a. Metrics for this
    # 3. Implement clustering [implemented, untested]
    # 4. Metrics to eval clusters
    # 5. Eval vs. known functional genes

import pipelines.utils
from pipelines.d2v_bigbird_base import TransformerClusterModel

#from bigbird.core import utils
#from bigbird.core import modeling
from bigbird.core import decoder
from bigbird.core import flags

from absl import app, logging

import tensorflow.compat.v2 as tf
#import tensorflow as tf
from tqdm import tqdm

import sys
import os
import time

import pickle

# from pubmed.ipynb: create container
# not sure what it does but it somehow, um, contains the model
from tensorflow.python.ops.variable_scope import EagerVariableStore
container = EagerVariableStore()

# Set flags, slightly modified from pumbed.ipynb
FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

# TODO: load flags from summarization.config

FLAGS.data_dir = "/fs/nexus-scratch/rhaworth/hmp-mini/"
FLAGS.output_dir = "/fs/nexus-scratch/rhaworth/output/"
FLAGS.attention_type = "block_sparse"
FLAGS.couple_encoder_decoder = True
FLAGS.max_encoder_length = 4096
FLAGS.max_decoder_length = 4096
FLAGS.block_size = 64
FLAGS.learning_rate = 1e-5
FLAGS.num_train_steps = 10000
FLAGS.attention_probs_dropout_prob = 0.0
FLAGS.hidden_dropout_prob = 0.0
FLAGS.use_gradient_checkpointing = True
FLAGS.vocab_model_file = "8mers" # currently only option
FLAGS.label_smoothing = 0.0 # imo it doesn't make sense to use label smoothing atm

# regulate architecture size / memory usage
# these settings allow us to use batch_size = 2, cutting training time in half
FLAGS.hidden_size = 384
FLAGS.intermediate_size = 1536
FLAGS.num_hidden_layers = 6

# only used by TPUEstimator implementation
FLAGS.train_batch_size = 4 
FLAGS.eval_batch_size = 4 
FLAGS.do_eval = True
FLAGS.do_export = True

# Init params, model, config
# I used to have a utils.BigBirdConfig() here but I prefer the flags, dropping that if possible

config = pipelines.utils.flags_as_dictionary()

with container.as_default():
    model = TransformerClusterModel(config)

# load weights
# TODO: make this a parameter or something
weights_filepath = os.path.join(FLAGS.output_dir, "epoch-9900.ckpt")
with open(weights_filepath, 'rb') as f:
  model.set_weights(pickle.load(f))

# compute embedding and return
@tf.function(experimental_compile=True, reduce_retracing=True)
def embed_only(features):
  embedding, _ = model._encode(features, training=False) # i think we can leave this false?
  return embedding

batch_size = 1

train_input_fn = pipelines.utils.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        is_training=True) # changed to true
dataset = train_input_fn({'batch_size': batch_size})

#eval_llh = tf.keras.metrics.Mean(name='eval_llh')

# k-means clustering
# TODO: make parameters
# final mini batch size = mini_batch_scale * batch_size
mini_batch_scale = 64
iterations = 10
k = 10
buffer = None
centroids = None

@tf.function
def print_tensor(t):
    tf.print(t, summarize=-1)


@tf.function(experimental_compile=True)
def fwd_only(features, labels):
  (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=False)
  return llh, logits, pred_ids
# populate centroids
for ex in tqdm(dataset.take(k//batch_size), position=0):
  if centroids == None:
    centroids = embed_only(ex)
  else:
    centroids = tf.concat([centroids, embed_only(ex)], 0)
  
  # inspect value
  _, _, ids = fwd_only(ex, ex)
  #logits = model.embeder.linear(c)
  #ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
  print("ids =")
  print_tensor(ids)

# check initial centroid shape
print(centroids.shape)

# run k-means training
# TODO: figure out if we should move some of this to a new @tf.function()
# (add 1 to take() to ensure we run the last batch)
for i, ex in enumerate(tqdm(dataset.take(mini_batch_scale*iterations + 1), position=0)):
  if i % mini_batch_scale == 0:
    if i != 0:
      print("iteration", i / (batch_size * mini_batch_scale))
      # calculate Euclidean distance between each row in buffer and each centroid
      centroid_dists = tf.norm(
        tf.reshape(buffer, (1, mini_batch_scale*batch_size, FLAGS.max_encoder_length, FLAGS.hidden_size)) -
        tf.reshape(centroids, (k, 1, FLAGS.max_encoder_length, FLAGS.hidden_size)),
        ord='euclidean', axis=[2,3]
      )

      # find min of each column of the distance matrix, generating the index of the nearest centroid
      nearest_centroids = tf.math.argmin(centroid_dists, axis=0)

      # compute new centroids
      new_centroids = []
      for i in range(0, k):
        # get cluster members, remove the 2nd axis that it adds for some reason
        cluster_members = tf.squeeze(tf.gather(buffer, tf.where(tf.equal(nearest_centroids, i)), axis=0), axis=1)
        # get current centroid, add axis for concat
        # including the current centroid in the mean gives it some weight when very few similar data points are present
        centroid = tf.reshape(tf.gather(centroids, i), [1, FLAGS.max_encoder_length, FLAGS.hidden_size])
        
        # compute new centroid, add to list
        new_centroid = tf.math.reduce_mean(tf.concat([cluster_members, centroid], 0), axis=0)
        new_centroids.append(new_centroid)

        # compute cluster distance
        print("mean distance for cluster", i, "=", tf.math.reduce_mean(
          tf.norm(cluster_members - centroid, ord='euclidean', axis=[1,2])
        ).numpy())
      
      # update centroid tensor
      centroids = tf.stack(new_centroids)
    # reset buffer
    buffer = embed_only(ex)
  else:
    # if we haven't reached the desired batch size, populate buffer
    buffer = tf.concat([buffer, embed_only(ex)], 0)
    
# record centroids to file
with open(os.path.join(FLAGS.output_dir, 'centroids.pickle'), 'wb') as f:
  pickle.dump(centroids, f)


# try to decode
full_mask = tf.ones([1, FLAGS.max_encoder_length])
tokenizer = pipelines.utils.D2v8merTokenizer(FLAGS.vocab_model_file)
for c in centroids:
  # ok i think this is how we decode only
  c = tf.reshape(c, [1, FLAGS.max_encoder_length, FLAGS.hidden_size])
  decoder_mask = decoder.create_self_attention_mask(FLAGS.max_encoder_length)
  outputs = model.decoder(c, decoder_mask, c, full_mask, training=False)

  logits = model.embeder.linear(outputs)
  ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

  print("ids =", ids)
  #print_tensor(ids)
  kmers = tokenizer.detokenize(ids)
  print("kmers =", kmers)
  #print_tensor(kmers)

  print("for just the original centroid,")
  logits = model.embeder.linear(c)
  ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

  print("ids =", ids)
  #print_tensor(ids)
  kmers = tokenizer.detokenize(ids)
  print("kmers =", kmers)
  #print_tensor(kmers)


# TODO:
  # decode centroids into sequences, see how well that even goes
  # traverse dataset and get list of points closest to each cluster, record those to file
    # possibly add yet another file that does that


# TODO: write a script to convert to TFRecord instead of whatever hack i'm doing

# this implementation traverses the whole dataset deterministically
# to shuffle instead, just set is_training = True and (optionally?) use .take()
# i think we shouldn't even be doing this for clustering lol
# TODO: run this with eval dataset on the whole dataset after clustering
#for i, ex in enumerate(tqdm(eval_dataset, position=0)):
#  embed = embed_only(ex)

  # TODO: store embeddings somehow, calculate kmeans

  #eval_llh(llh)
#print('Log Likelihood = {}'.format(eval_llh.result().numpy()))

# to get predictions, call:
#_, _, pred_ids = fwd_only(ex[0], ex[1])
# then detokenize pred_ids

def main(_):
  print("Running main")

if __name__ == '__main__':
  #tf.compat.v1.disable_v2_behavior()
  tf.compat.v1.enable_resource_variables()
  app.run(main)
