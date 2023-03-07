# Reusable helper functions for impementing pipelines

import os
import sys
import numpy as np
import tensorflow as tf
#from dna2vec.dna2vec.multi_k_model import MultiKModel
from gensim.models import word2vec

from pipelines.d2v_bigbird_base import TransformerClusterModel

from bigbird.core import modeling
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
        # new implementation, just opens file and extracts string:
        # okay this is working properly
        return tf.io.read_file(record)

    def _tokenize_contig(contig):
        # Initialize tokenizer
        # TODO: test whether tft.SentencepieceTokenizer can work with dna2vec somehow
            # I don't think so
        # Also is this being called repeatedly? I really hope not or I have to fix that
        tokenizer = D2v8merTokenizer(vocab_model_file)

        # For this implementation, we use the same contig for input and output
        print("tokenizing contig: ", contig)
        contig_ids = tokenizer.tokenize(contig)
        if isinstance(contig_ids, tf.RaggedTensor):
            contig_ids = contig_ids.to_tensor(0)
        contig_ids = contig_ids[:max_encoder_length]

        print("tokenized: ", contig_ids)

        return contig_ids

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # Load dataset from text files
        input_files = tf.io.gfile.glob(
            os.path.join(data_dir, "*.txt"))
        
        print("first input file: ", input_files[0])

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            print("training")
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.shuffle(buffer_size=len(input_files))

            # Non deterministic mode means that the interleaving is not exact.
            # This adds even more randomness to the training pipeline.
            #d = d.interleave(tf.data.TFRecordDataset,
            #                    deterministic=False,
            #                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            d = tf.data.TFRecordDataset(input_files)

        """
        for ex in d.take(1):
            print(ex)
            decoded = _decode_record(ex)
            tokenized = _tokenize_contig(decoded)
            print(tokenized)
        """

        # _decode_record currently breaks b/c D2v8merTokenizer is expecting a string, not a TF example or whatever
        # and i guess we don't really have a record but a bunch of text files
        # commenting out for now, may try to get this working in the future if it's helpful
        d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

        d = d.map(_tokenize_contig,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=is_training)
        print("made it here, code breaks at padded_batch now lol")
        if is_training:
            d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            d = d.repeat()
        d = d.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
                            drop_remainder=True)  # For static shape
        return d

    return input_fn

# Define model_fn_builder, modified from run_summarization.py
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

    # Use custom loss function
    total_loss = loss_fn(logits, labels)

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

# Define loss function
def loss_fn(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits, labels)

# Define dna2vec tokenize/detokenize wrapper
class D2v8merTokenizer():
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

# TODO: vector-based clustering
# god why is this so fucking hard
# okay i guess i'll just try to implement like, whichever one is the easiest. probably k-means.
# then i need a function to find the centroid.
# maybe test everything else first?