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
from bigbird.core import flags

from absl import app, logging

#import tensorflow.compat.v2 as tf
import tensorflow as tf
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

train_input_fn = pipelines.utils.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        is_training=True) # changed to true
dataset = train_input_fn({'batch_size': 2})

#eval_llh = tf.keras.metrics.Mean(name='eval_llh')

# k-means clustering
# TODO: make parameters
# final mini batch size = mini_batch_scale * batch_size
mini_batch_scale = 256
iterations = 10
k = 10
buffer = None
centroids = None

# populate centroids
for ex in tqdm(dataset.take(k), position=0):
  if centroids == None:
    centroids = embed_only(ex)
  else:
    centroids = tf.concat([centroids, ex], 0)

# check initial centroids
print(centroids)
print(centroids.shape)

# run k-means training
# TODO: figure out if we should move some of this to an @tf.function()
# (add 1 to take() to ensure we run the last batch)
for i, ex in enumerate(tqdm(dataset.take(2*mini_batch_scale*iterations + 1), position=0)):
  if i % mini_batch_scale == 0:
    if i != 0:
      # calculate Euclidean distance between each row in buffer and each centroid
      centroid_dists = tf.norm(
        tf.reshape(buffer, (1, mini_batch_scale*2, FLAGS.max_encoder_length, FLAGS.hidden_size)) -
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
        centroid = tf.reshape(tf.gather(centroids, i), [1, FLAGS.max_encoder_length, FLAGS.hidden_size])
        
        # compute new centroid, add to list
        new_centroid = tf.math.reduce_mean(tf.concat([cluster_members, centroid], 0), axis=0)
        new_centroids.append(new_centroid)
      
      # update centroid tensor
      centroids = tf.stack(new_centroids)
    # reset buffer
    buffer = embed_only(ex)
  else:
    # if we haven't reached the desired batch size, populate buffer
    buffer = tf.concat([buffer, embed_only(ex)], 0)
    
    


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