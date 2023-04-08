# Reusable helper functions for impementing pipelines

import os
import sys
#import numpy as np
import tensorflow as tf
from absl import logging
import json

from pipelines.d2v_bigbird_base import TransformerClusterModel

#from bigbird.core import modeling
from bigbird.core import utils
from bigbird.core import optimization
from bigbird.core import flags

# Define flags (from run_summarization)

## Required parameters

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "tfds://scientific_papers/pubmed",
    "The input data dir. Should contain the TFRecord files. "
    "Can be TF Dataset with prefix tfds://")

flags.DEFINE_string(
    "output_dir", "/tmp/bigb",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BigBird model).")

flags.DEFINE_integer(
    "max_encoder_length", 128,
    "The maximum total input sequence length after SentencePiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_decoder_length", 128,
    "The maximum total input sequence length after SentencePiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "substitute_newline", None,
    "Replace newline charachter from text with supplied string.")

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training.")

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model as TF SavedModel.")

flags.DEFINE_integer(
    "train_batch_size", 8,
    "Local batch size for training. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_integer(
    "eval_batch_size", 8,
    "Local batch size for eval. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_string(
    "optimizer", "Adafactor",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.")

flags.DEFINE_float(
    "learning_rate", 0.32,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 1000,
    "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 100,
    "Number of steps to perform linear warmup.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 2000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "max_eval_steps", 100,
    "Maximum number of eval steps.")

flags.DEFINE_bool(
    "couple_encoder_decoder", False,
    "Whether to tie encoder and decoder weights.")

flags.DEFINE_integer(
    "beam_size", 5,
    "Beam size for decoding.")

flags.DEFINE_float(
    "alpha", 0.8,
    "Strength of length normalization for beam search.")

flags.DEFINE_float(
    "label_smoothing", 0.1,
    "Label smoothing for prediction cross entropy loss.")

# TODO: implement reverse complement w/ lexicographical comparison

# custom flags-to-dictionary function
def flags_as_dictionary():
    """Get current config from flag."""

    # TODO: adjust size of vocab depending on kmer embedding
    vocab_size = 65536

    # Resolve vocab file location from hotword
    if FLAGS.vocab_model_file == "8mers":
        FLAGS.vocab_model_file = str(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),
                                                  'dna2vec/pretrained/dna2vec-8mer-only.w2v'))

    config = {
        # transformer basic configs
        "attention_probs_dropout_prob": FLAGS.attention_probs_dropout_prob,
        "hidden_act": FLAGS.hidden_act,
        "hidden_dropout_prob": FLAGS.hidden_dropout_prob,
        "hidden_size": FLAGS.hidden_size,
        "initializer_range": FLAGS.initializer_range,
        "intermediate_size": FLAGS.intermediate_size,
        "max_position_embeddings": FLAGS.max_position_embeddings,
        "num_attention_heads": FLAGS.num_attention_heads,
        "num_hidden_layers": FLAGS.num_hidden_layers,
        "type_vocab_size": FLAGS.type_vocab_size,
        "scope": FLAGS.scope,
        "use_bias": FLAGS.use_bias,
        "rescale_embedding": FLAGS.rescale_embedding,
        "use_gradient_checkpointing": FLAGS.use_gradient_checkpointing,
        "vocab_model_file": FLAGS.vocab_model_file,
        # sparse mask configs
        "attention_type": FLAGS.attention_type,
        "norm_type": FLAGS.norm_type,
        "block_size": FLAGS.block_size,
        "num_rand_blocks": FLAGS.num_rand_blocks,
        # common bert configs
        "data_dir": FLAGS.data_dir,
        "output_dir": FLAGS.output_dir,
        "init_checkpoint": FLAGS.init_checkpoint,
        "max_encoder_length": FLAGS.max_encoder_length,
        "substitute_newline": FLAGS.substitute_newline,
        "do_train": FLAGS.do_train,
        "do_eval": FLAGS.do_eval,
        "do_export": FLAGS.do_export,
        "train_batch_size": FLAGS.train_batch_size,
        "eval_batch_size": FLAGS.eval_batch_size,
        "optimizer": FLAGS.optimizer,
        "learning_rate": FLAGS.learning_rate,
        "num_train_steps": FLAGS.num_train_steps,
        "num_warmup_steps": FLAGS.num_warmup_steps,
        "save_checkpoints_steps": FLAGS.save_checkpoints_steps,
        "weight_decay_rate": FLAGS.weight_decay_rate,
        "optimizer_beta1": FLAGS.optimizer_beta1,
        "optimizer_beta2": FLAGS.optimizer_beta2,
        "optimizer_epsilon": FLAGS.optimizer_epsilon,
        # TPU settings
        "use_tpu": FLAGS.use_tpu,
        "tpu_name": FLAGS.tpu_name,
        "tpu_zone": FLAGS.tpu_zone,
        "tpu_job_name": FLAGS.tpu_job_name,
        "gcp_project": FLAGS.gcp_project,
        "master": FLAGS.master,
        "num_tpu_cores": FLAGS.num_tpu_cores,
        "iterations_per_loop": FLAGS.iterations_per_loop,
    }

    # pretraining dedicated flags
    if hasattr(FLAGS, "max_predictions_per_seq"):
        config["max_predictions_per_seq"] = FLAGS.max_predictions_per_seq
    if hasattr(FLAGS, "masked_lm_prob"):
        config["masked_lm_prob"] = FLAGS.masked_lm_prob
    if hasattr(FLAGS, "max_eval_steps"):
        config["max_eval_steps"] = FLAGS.max_eval_steps
    if hasattr(FLAGS, "preprocessed_data"):
        config["preprocessed_data"] = FLAGS.preprocessed_data
    if hasattr(FLAGS, "use_nsp"):
        config["use_nsp"] = FLAGS.use_nsp

    # classifier dedicated flags
    if hasattr(FLAGS, "num_labels"):
        config["num_labels"] = FLAGS.num_labels

    # summarization dedicated flags
    if hasattr(FLAGS, "max_decoder_length"):
        config["max_decoder_length"] = FLAGS.max_decoder_length
    if hasattr(FLAGS, "trainable_bias"):
        config["trainable_bias"] = FLAGS.trainable_bias
    if hasattr(FLAGS, "couple_encoder_decoder"):
        config["couple_encoder_decoder"] = FLAGS.couple_encoder_decoder
    if hasattr(FLAGS, "beam_size"):
        config["beam_size"] = FLAGS.beam_size
    if hasattr(FLAGS, "alpha"):
        config["alpha"] = FLAGS.alpha
    if hasattr(FLAGS, "label_smoothing"):
        config["label_smoothing"] = FLAGS.label_smoothing

    # calculate vocab
    config["vocab_size"] = vocab_size

    return config

# Define input_fn_builder, modified from run_summarization.py
# If this has issues, just double return everything
def input_fn_builder(data_dir, vocab_model_file, max_encoder_length,
                     max_decoder_length, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        # old implementation:
        """
        name_to_features = {
            "contig": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, name_to_features)
        return example["contig"]
        """
        # we aren't working with TFRecord data, so simply read file
        return tf.io.read_file(record)

    def _tokenize_contig(contig):
        # initialize tokenizer
        tokenizer = D2v8merTokenizer(vocab_model_file)

        # use the same contig for input and output
        contig_ids = tokenizer.tokenize(contig)
        if isinstance(contig_ids, tf.RaggedTensor):
            contig_ids = contig_ids.to_tensor(0)
        contig_ids = contig_ids[:max_encoder_length]

        # ideally, avoid returning the same thing twice
        return contig_ids#, contig_ids

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # Load dataset from text files
        input_files = tf.io.gfile.glob(
            os.path.join(data_dir, "*.txt"))
        
        # because we aren't using a TFRecord, this needs to happen whether we're training or not
        # otherwise, it would be in the is_training block
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=len(input_files))

            # Non deterministic mode means that the interleaving is not exact.
            # This adds even more randomness to the training pipeline.
            #d = d.interleave(tf.data.TFRecordDataset,
            #                    deterministic=False,
            #                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # this breaks everything unless we convert to TFRecord
        #else:
        #    d = tf.data.TFRecordDataset(input_files)

        d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

        d = d.map(_tokenize_contig,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=is_training)
        
        if is_training:
            d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            d = d.repeat()
        
        # this originally had padded_shape = ([max_encoder_length], [max_decoder_length])
        # but we're only returning one thing, so drop decoder length
        d = d.padded_batch(batch_size, [max_encoder_length],
                            drop_remainder=True)  # For static shape
        return d

    return input_fn

# define model_fn_builder, modified from run_summarization.py
# NOTE: unused in current implementation
def model_fn_builder(transformer_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    # idk if we need this
    if isinstance(features, dict):
      if not labels and "target_ids" in features:
        labels = features["target_ids"]
      features = features["input_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Changed model
    model = TransformerClusterModel(transformer_config)
    (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=is_training)

    # idk how this was working without them before but we need the smoothing and vocab size vars
    total_loss = padded_cross_entropy_loss(
        logits, labels,
        transformer_config["label_smoothing"],
        transformer_config["vocab_size"])

    tvars = tf.compat.v1.trainable_variables()
    utils.log_variables(tvars, transformer_config["ckpt_var_list"])

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

        learning_rate = optimization.get_linear_warmup_rsqrt_decay_lr(
            init_lr=transformer_config["learning_rate"],
            hidden_size=transformer_config["hidden_size"],
            num_warmup_steps=transformer_config["num_warmup_steps"])

        optimizer = optimization.get_optimizer(transformer_config, learning_rate)

        global_step = tf.compat.v1.train.get_global_step()

        # TODO: decide if we're using position embedding
        #if not transformer_config["use_bias"]:
        #    logging.info("Fixing position embedding, i.e. not trainable.")
        #    posemb = "pegasus/embeddings/position_embeddings"
        #    tvars = list(filter(lambda v: v.name.split(":")[0] != posemb, tvars))

        gradients = optimizer.compute_gradients(total_loss, tvars)
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            host_call=utils.add_scalars_to_summary(
                transformer_config["output_dir"],
                {"learning_rate": learning_rate}))

    elif mode == tf.estimator.ModeKeys.EVAL:
        tokenizer = D2v8merTokenizer(transformer_config["vocab_model_file"])

        # TODO: define some kind of metric for evaluation, e.g. edit distance
        # Or I guess it would have to be alignment or something?
        # You know what maybe this whole part is too hard without external tools
        def eval_func(label_sent, pred_sent):
            return label_sent - pred_sent

        def metric_fn(loss, log_probs, label_ids, pred_ids):
            loss = tf.compat.v1.metrics.mean(values=loss)
            log_probs = tf.compat.v1.metrics.mean(
                values=log_probs,
                weights=tf.cast(tf.not_equal(label_ids, 0), tf.float32))
            metric_dict = {
                "prediction_loss": loss,
                "log_likelihood": log_probs,
            }

            if not transformer_config["use_tpu"]:
                # Approximate ROUGE scores if not running on tpus.
                # Always run externally for final scores.
                label_sent = tokenizer.detokenize(label_ids)
                pred_sent = tokenizer.detokenize(pred_ids)
                
                # TODO: make this use whatever eval score instead of ROUGE score
                rouge_value = tf.compat.v1.py_func(
                    func=eval_func,
                    inp=[label_sent, pred_sent],
                    Tout=[tf.float64, tf.float64, tf.float64],
                    stateful=False)
                rouge_value = tf.cast(rouge_value, tf.float32)
                rouge1 = tf.compat.v1.metrics.mean(values=rouge_value[0])
                rouge2 = tf.compat.v1.metrics.mean(values=rouge_value[1])
                rougeL = tf.compat.v1.metrics.mean(values=rouge_value[2])  # pylint: disable=invalid-name

                metric_dict.update({
                    "eval/Rouge-1": rouge1,
                    "eval/Rouge-2": rouge2,
                    "eval/Rouge-L": rougeL,
                })
            return metric_dict

        eval_metrics = (metric_fn,
                        [total_loss, llh, labels, pred_ids])
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics)
    else:
        prediction_dict = {"pred_ids": pred_ids}
        if not transformer_config["use_tpu"]:
            tokenizer = D2v8merTokenizer()

            pred_sent = tokenizer.detokenize(pred_ids)

            prediction_dict.update({"pred_sent": pred_sent})

        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=prediction_dict)

    return output_spec

  return model_fn

def serving_input_fn_builder(batch_size, max_encoder_length,
                             vocab_model_file):
  """Creates an `input_fn` closure for exported SavedModel."""
  def dynamic_padding(inp, min_size):
    pad_size = tf.maximum(min_size - tf.shape(inp)[1], 0)
    paddings = [[0, 0], [0, pad_size]]
    return tf.pad(inp, paddings)

  def input_fn():
    # sequence input
    seq = tf.compat.v1.placeholder(tf.string, [batch_size], name="input_seq")

    # tokenize sequence
    tokenizer = D2v8merTokenizer(vocab_model_file)
    ids = tokenizer.tokenize(seq)
    if isinstance(ids, tf.RaggedTensor):
      ids = ids.to_tensor(0)

    # padding: Pad only if necessary and reshape properly
    # TODO: fix this part
    # actually it might work fine bc it's expecting only one set of ids
    # TODO: see if this part needs to be fixed
    padded_ids = dynamic_padding(ids, max_encoder_length)
    ids = tf.slice(padded_ids, [0, 0], [batch_size, max_encoder_length])

    receiver_tensors = {"input": seq}
    features = {"input_ids": tf.cast(ids, tf.int32, name="input_ids")}

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=receiver_tensors)

  return input_fn

# from run_summarization.py
# i was just omitting all the smoothing and stuff but i need to handle the padding
def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  with tf.name_scope("loss"):

    if labels is not None:
      # Calculate smoothing cross entropy
      with tf.name_scope("smoothing_cross_entropy"):
        confidence = 1.0 - smoothing
        vocab_float = tf.cast(vocab_size - 1, tf.float32)
        low_confidence = (1.0 - confidence) / vocab_float
        soft_targets = tf.one_hot(
            labels,
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=soft_targets)

        # Calculate the best (lowest) possible value of cross entropy, and
        # subtract from the cross entropy loss.
        normalizing_constant = -(
            confidence * tf.math.log(confidence) + vocab_float *
            low_confidence * tf.math.log(low_confidence + 1e-20))
        xentropy -= normalizing_constant

      weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
      loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    else:
      loss = tf.constant(0.0)

    return loss

# dna2vec tokenizer/detokenizer only
class D2v8merTokenizer():
    def __init__(self, filepath):
        # read file, split into lines, and drop first and last line
        vocab = tf.strings.split(tf.io.read_file(filepath), sep='\r\n')[1:-1]

        # grab list of kmers
        kmers = tf.strings.substr(vocab, 0, 8)
        
        # make hashmap to convert kmers into ids
        # TODO: read in vocab size from file
        self.kmers_to_ids = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(kmers, tf.range(65536)),
            default_value=-1
        )

        # make reverse hashmap to convert ids back to kmers
        self.ids_to_kmers = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.range(65536), kmers),
            default_value='NNNNNNNN'
        )

    def tokenize(self, seq):
        if seq.shape != ():
            raise ValueError("Expected scalar tensor, got ", seq)
        # get number of kmers then decompose
        num_kmers = tf.strings.length(seq) - 7
        kmers = tf.strings.substr(seq, tf.range(num_kmers), tf.fill([num_kmers], 8))
        # return ids
        return self.kmers_to_ids.lookup(kmers)

    def detokenize(self, ids):
        # reconstruct sequence from kmers
        # the difficult part here is that there might be conflicts between kmers
        # TODO: handle those somehow
            # idea: use kmers to select nucleotide at each index with highest support
            # e.g. if there are 2 overlapping kmers, pick randomly
                # if there are 5 = [A,C,A,G,T], pick A
                # if there are 8 = [A,A,A,C,C,C,G,T], pick randomly between A and C
        kmers = self.ids_to_kmers.lookup(ids)
        # for now, just literally join all the kmers
        return tf.strings.reduce_join(kmers)

# dna2vec tokenize/detokenize wrapper with embeddings
# TODO: make this work and possibly convert into EmbeddingLayer
class D2v8merTokenizerEmbeder():
    def __init__(self, filepath):
        # read file, split into lines, and drop first and last line
        vocab = tf.strings.split(tf.io.read_file(filepath), sep='\r\n')[1:-1]

        # split into kmers and embedding table; convert embedding table to 2d float32 tensor and set shape
        kmers = tf.strings.substr(vocab, 0, 8)
        self.embed_table = tf.strings.split(tf.strings.substr(vocab, 9, 2000), ' ')
        self.embed_table = tf.strings.to_number(self.embed_table, tf.float32)
        self.embed_table = tf.reshape(self.embed_table, [65536, 100])

        # make hashmap to convert kmers into ids
        self.kmer_map = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(kmers, tf.range(65536)),
            default_value=-1
        )

    def tokenize(self, seq):
        # assume seq is a scalar tensor; TODO: check
        if seq.shape != ():
            raise ValueError("Expected scalar tensor, got ", seq)
        # get number of kmers then decompose
        num_kmers = tf.strings.length(seq) - 7
        kmers = tf.strings.substr(seq, tf.range(num_kmers), tf.fill([num_kmers], 8))
        # lookup ids
        ids = self.kmer_map.lookup(kmers)
        # return embedding
        return tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embed_table, ids)

    def detokenize(self, embed):
        # The difficult part here is that there may be conflicts
        # TODO: handle those somehow
        # also TODO: make this actually work lol, basically just re-implement similar_by_vector
        # For now, just fetch 8-mers
        seq = ''
        for i in range(0, embed.shape(0), 8):
            tmp = self.model.similar_by_vector(embed[i], topn=1)
            seq += tmp[0]
        # NOTE: this also drops the last few nucleotides if len % 8 != 0, but w/e it's not a good solution anyway
        return seq

# Define dna2vec embedding layer class
# Base implementation with no positional information beyond 8-mers
# TODO: figure out if we should do this instead lol
class D2vKmerEmbeddingLayer(tf.keras.layers.Layer):
    """An embedding layer."""

    def __init__(self,
                initializer,
                scale_emb=False,
                name="embeddings"):
        super(D2vKmerEmbeddingLayer, self).__init__(name=name)
        # Vocab size should always be 65,536, embedding dim should always be 100
        self.vocab_size = 65536
        self.emb_dim = 100
        self.scale_emb = scale_emb

        with tf.compat.v1.variable_scope(name):
            self.word_embeddings = tf.compat.v1.get_variable(
                "word_embeddings", [self.vocab_size, self.emb_dim],
                dtype=tf.float32, initializer=initializer)

    def call(self,
            input_ids,
            seq_length,
            start_pos=0,
            training=None):
        if input_ids is None:
            return None

        # subtoken embedding
        output = tf.nn.embedding_lookup(params=self.word_embeddings, ids=input_ids)

        if self.scale_emb:
            output = output * self.emb_dim ** 0.5
        
        return output

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
        x: A float32 tensor with shape [..., hidden_size]
        Returns:
        float32 tensor with shape [..., vocab_size].
        """
        with tf.compat.v1.name_scope("presoftmax_linear"):
            logits = tf.tensordot(x, self.word_embeddings, [[-1], [1]])
        return logits

# custom flag save function
def save_flags(path):
  """Save current flag config."""
  config = flags_as_dictionary()
  with tf.io.gfile.GFile(path, "w") as f:
    json.dump(config, f, indent=4, sort_keys=True)

  # log flags
  max_len = max([len(ii) for ii in config.keys()])
  fmt_string = "\t%" + str(max_len) + "s : %s"
  logging.info("Arguments:")
  for key, value in sorted(config.items()):
    logging.info(fmt_string, key, value)

  return config

# TODO: add anything needed for clustering
# so far it all seems fine in cluster-eval.py but it might be best to add a dedicated layer or smth