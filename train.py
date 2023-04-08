# Train model

import pipelines.utils
from pipelines.d2v_bigbird_base import TransformerClusterModel

#from bigbird.core import utils
#from bigbird.core import modeling
from bigbird.core import flags

from absl import app #, logging

#import tensorflow.compat.v2 as tf
import tensorflow as tf
from tqdm import tqdm

import sys
import os
import time

# alright whatever i'm gonna store things myself watch this
import pickle

# from pubmed.ipynb: create container
# not sure what it does but it somehow, um, contains the model
from tensorflow.python.ops.variable_scope import EagerVariableStore
container = EagerVariableStore()

# set flags, slightly modified from pumbed.ipynb
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
FLAGS.hidden_size = 384 # must cleanly divide 768
FLAGS.intermediate_size = 1536
#FLAGS.num_hidden_layers = 6
FLAGS.num_attention_heads = 6

# only used by TPUEstimator implementation
FLAGS.train_batch_size = 4 
FLAGS.eval_batch_size = 4 
FLAGS.do_eval = True
FLAGS.do_export = True

# finalize config, create model
config = pipelines.utils.flags_as_dictionary()

with container.as_default():
    model = TransformerClusterModel(config)

# call input and model function builders
train_input_fn = pipelines.utils.input_fn_builder(
    data_dir=FLAGS.data_dir,
    vocab_model_file=FLAGS.vocab_model_file,
    max_encoder_length=FLAGS.max_encoder_length,
    max_decoder_length=FLAGS.max_decoder_length,
    is_training=True)
# set as large as possible, limited by memory
dataset = train_input_fn({'batch_size': 2})

# define training function w/ backpropogation
@tf.function(experimental_compile=True, reduce_retracing=True)
def fwd_bwd(features, labels):
  with tf.GradientTape() as g:
    (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=True)
    loss = pipelines.utils.padded_cross_entropy_loss(
      logits, labels, 
      config["label_smoothing"], config["vocab_size"])
  grads = g.gradient(loss, model.trainable_weights)
  return loss, llh, logits, pred_ids, grads

# create output directory
tf.io.gfile.makedirs(FLAGS.output_dir)

# save flags
if FLAGS.do_train:
  pipelines.utils.save_flags(os.path.join(FLAGS.output_dir, "summarization.config"))

# train model
opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)
train_loss = tf.keras.metrics.Mean(name='train_loss')
for i, ex in enumerate(tqdm(dataset.take(FLAGS.num_train_steps), position=0)):
  # ex, ex instead of ex[0], ex[1] since we only have one set of ids
  loss, llh, logits, pred_ids, grads = fwd_bwd(ex, ex)
  opt.apply_gradients(zip(grads, model.trainable_weights))
  train_loss(loss)
  if i % 10 == 0:
    print('Loss = {} '.format(train_loss.result().numpy()))
  if i % 100 == 0 and i != 0:
    out_path = os.path.join(FLAGS.output_dir, 'epoch-' + str(i) + '.pickle')
    with open(out_path, 'wb') as f:
      pickle.dump(model.get_weights(), f)
    print("Saved weights to", out_path)

# final save
out_path = os.path.join(FLAGS.output_dir, 'epoch-last.pickle')
with open(out_path, 'wb') as f:
  pickle.dump(model.get_weights(), f)
print("Saved weights to", out_path)

print("Training complete. Evaluating.")

# define forward pass only for eval
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


# TPUEstimator implementation
# from run_summarization.py, modified to use functions from pipelines.utils
# this seems to mysteriously freeze, so i'm reverting to the old implementation
def main(_):
  """
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_eval` must be True.")

  transformer_config = pipelines.utils.flags_as_dictionary()

  if FLAGS.max_encoder_length > transformer_config["max_position_embeddings"]:
    raise ValueError(
        "Cannot use sequence length %d because the model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_encoder_length,
         transformer_config["max_position_embeddings"]))

  tf.io.gfile.makedirs(FLAGS.output_dir)
  if FLAGS.do_train:
    pipelines.utils.save_flags(os.path.join(FLAGS.output_dir, "summarization.config"))

  model_fn = pipelines.utils.model_fn_builder(transformer_config)
  estimator = utils.get_estimator(transformer_config, model_fn)
  #tmp_data_dir = os.path.join(FLAGS.output_dir, "tfds")

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", estimator.train_batch_size)
    logging.info("  Num steps = %d", FLAGS.num_train_steps)
    train_input_fn = pipelines.utils.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    logging.info("  Batch size = %d", estimator.eval_batch_size)

    eval_input_fn = pipelines.utils.input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
        is_training=False)

    # Run continuous evaluation for latest checkpoint as training progresses.
    last_evaluated = None
    while True:
      latest = tf.train.latest_checkpoint(FLAGS.output_dir)
      if latest == last_evaluated:
        if not latest:
          logging.info("No checkpoints found yet.")
        else:
          logging.info("Latest checkpoint %s already evaluated.", latest)
        time.sleep(300)
        continue
      else:
        logging.info("Evaluating check point %s", latest)
        last_evaluated = latest

        current_step = int(os.path.basename(latest).split("-")[1])
        output_eval_file = os.path.join(
            FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
        result = estimator.evaluate(input_fn=eval_input_fn,
                                    steps=FLAGS.max_eval_steps,
                                    checkpoint_path=latest)

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_export:
    logging.info("***** Running export *****")

    serving_input_fn = pipelines.utils.serving_input_fn_builder(
        batch_size=FLAGS.eval_batch_size,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length)

    estimator.export_saved_model(
        os.path.join(FLAGS.output_dir, "export"), serving_input_fn)
    """

if __name__ == '__main__':
  #tf.compat.v1.disable_v2_behavior()
  tf.compat.v1.enable_resource_variables()
  app.run(main)