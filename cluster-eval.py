# TODO:
    # 1. Import saved model weights
        # done, untested
    # 2. Predict for arbitrary string with saved model; should produce the same string or very similar
        # 2a. Metrics for this
    # 3. Implement clustering
    # 4. Metrics to eval clusters
    # 5. Eval vs. known functional genes

import pipelines.utils
from pipelines.d2v_bigbird_base import TransformerClusterModel

#from bigbird.core import utils
#from bigbird.core import modeling
from bigbird.core import flags

from absl import app, logging

import tensorflow.compat.v2 as tf
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
FLAGS.num_attention_heads = 6

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

# forward pass only for eval
@tf.function(experimental_compile=True)
def fwd_only(features, labels):
  (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=False)
  return llh, logits, pred_ids

eval_input_fn = pipelines.utils.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        is_training=False)
eval_dataset = eval_input_fn({'batch_size': 2})

eval_llh = tf.keras.metrics.Mean(name='eval_llh')

for ex in tqdm(eval_dataset, position=0):
  llh, logits, pred_ids = fwd_only(ex, ex)
  eval_llh(llh)
print('Log Likelihood = {}'.format(eval_llh.result().numpy()))

# to get predictions, call:
#_, _, pred_ids = fwd_only(ex[0], ex[1])
# then detokenize pred_ids

def main(_):
  print("Running main")

if __name__ == '__main__':
  #tf.compat.v1.disable_v2_behavior()
  tf.compat.v1.enable_resource_variables()
  app.run(main)