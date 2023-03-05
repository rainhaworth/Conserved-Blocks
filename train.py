# Train model

import pipelines.utils
from pipelines.d2v_bigbird_base import TransformerClusterModel

#from bigbird.core import utils
#from bigbird.core import modeling
from bigbird.core import flags

from absl import app

import tensorflow.compat.v2 as tf
from tqdm import tqdm
import sys

# From ipynb: create container
# not sure what it does but it somehow, um, contains the model
from tensorflow.python.ops.variable_scope import EagerVariableStore
container = EagerVariableStore()

# Set flags, slightly modified from pumbed.ipynb
FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

# it keeps yelling at me about the flags
# do i have to just define them all?
# i think i should put this on the cluster

# data_dir should probably be a parameter
FLAGS.data_dir = "./data-tmp/"
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
# idk if i need vocab_model_file at all? leaving it out for now
#FLAGS.vocab_model_file = "gpt2"

print(flags.as_dictionary())

# Init params, model, config
# I used to have a utils.BigBirdConfig() here but I prefer the flags, dropping that if possible

config = flags.as_dictionary()

with container.as_default():
    model = TransformerClusterModel(config)

# Call input and model function builders
train_input_fn = pipelines.utils.input_fn_builder(
    data_dir=FLAGS.data_dir,
    vocab_model_file=FLAGS.vocab_model_file,
    max_encoder_length=FLAGS.max_encoder_length,
    max_decoder_length=FLAGS.max_decoder_length,
    is_training=True)
# Idk if this is the right batch size to use but it's in the notebook
dataset = train_input_fn({'batch_size': 8})

model_fn = pipelines.utils.model_fn_builder(flags)

# Define forward + backward pass, idk if this is what we need to do lol but it's from the file
# I think this is just like, a training function with backpropogation (the gradient tape thing)
# And the eval function is just the model part
# Should this go in the utils file? possibly
@tf.function(experimental_compile=True)
def fwd_bwd(features, labels):
  with tf.GradientTape() as g:
    (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=True)
    loss = pipelines.utils.loss_fn(
        logits, labels) # Include transformer config if needed
  grads = g.gradient(loss, model.trainable_weights)
  return loss, llh, logits, pred_ids, grads

# inspect at a few examples
#for ex in dataset.take(3):
#  print(ex)

# check outputs
#loss, llh, logits, pred_ids, grads = fwd_bwd(ex[0], ex[1])
#print('Loss: ', loss)

# see ipynb if loading pretrained

# Train model
opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)
train_loss = tf.keras.metrics.Mean(name='train_loss')

# NOTE: the ex[0] and ex[1] should break; if the point of failure is here, change both to ex or ex[0]
  # but if the point of failure is somewhere in input_fn_builder, we can change that instead
for i, ex in enumerate(tqdm(dataset.take(FLAGS.num_train_steps), position=0)):
  loss, llh, logits, pred_ids, grads = fwd_bwd(ex[0], ex[1])
  opt.apply_gradients(zip(grads, model.trainable_weights))
  train_loss(loss)
  if i% 10 == 0:
    print('Loss = {} '.format(train_loss.result().numpy()))

print("Training done. Evaluating...")

# Eval code
# TODO: make new file or add a conditional or smth
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
eval_dataset = eval_input_fn({'batch_size': 8})

eval_llh = tf.keras.metrics.Mean(name='eval_llh')

for ex in tqdm(eval_dataset, position=0):
  llh, logits, pred_ids = fwd_only(ex[0], ex[1])
  eval_llh(llh)
print('Log Likelihood = {}'.format(eval_llh.result().numpy()))

# to get predictions, call:
#_, _, pred_ids = fwd_only(ex[0], ex[1])
# then detokenize pred_ids

# absl stuff
def main(_):
  print("Running absl")

if __name__ == '__main__':
  app.run(main)