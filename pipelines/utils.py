# Reusable helper functions for impementing pipelines

import os
import numpy as np
import tensorflow as tf
from dna2vec.dna2vec.multi_k_model import MultiKModel
from gensim.models import word2vec

from pipelines.d2v_bigbird_base import TransformerClusterModel

from bigbird.core import modeling
from bigbird.core import utils
from bigbird.core import optimization

# TODO: implement reverse complement w/ lexicographical comparison


# Define input_fn_builder, modified from run_summarization.py
# If this has issues, just double return everything
def input_fn_builder(data_dir, vocab_model_file, max_encoder_length,
                     max_decoder_length, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "contig": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, name_to_features)
        return example["contig"]

    def _tokenize_contig(contig):
        # Initialize tokenizer
        # TODO: test whether tft.SentencepieceTokenizer can work with dna2vec somehow
            # I don't think so
        # Also is this being called repeatedly? I really hope not or I have to fix that
        tokenizer = D2v8merTokenizer()

        # For this implementation, we use the same contig for input and output
        contig_ids = tokenizer.tokenize(contig)
        if isinstance(contig_ids, tf.RaggedTensor):
            contig_ids = contig_ids.to_tensor(0)
        contig_ids = contig_ids[:max_encoder_length]

        return contig_ids

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # Load dataset from text files
        input_files = tf.io.gfile.glob(
            os.path.join(data_dir, "*.txt"))

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.shuffle(buffer_size=len(input_files))

            # Non deterministic mode means that the interleaving is not exact.
            # This adds even more randomness to the training pipeline.
            d = d.interleave(tf.data.TFRecordDataset,
                                deterministic=False,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

        d = d.map(_tokenize_contig,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=is_training)

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
        tokenizer = D2v8merTokenizer()

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
    def __init__(self):
        # This makes it only work from the base directory (Conserved-Blocks)
        # Maybe make the directory a parameter?
        filepath = 'dna2vec/pretrained/dna2vec-8mer-only.w2v'
        self.model = word2vec.KeyedVectors.load_word2vec_format(filepath, binary=False)
    def tokenize(self, seq):
        length = len(seq) - 8 + 1
        embed = tf.zeros([length, 100])
        for i in range(length): # - 8 + 1
            embed[i] = self.model.vector(seq[i:i+8])
        return np.array(embed)

    def detokenize(self, embed):
        # The difficult part here is that there may be conflicts
        # TODO: handle those somehow
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