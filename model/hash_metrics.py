# metrics for learning to hash
import tensorflow as tf

# add loss/metric hyperparameters here
def hash_metric_factory(h=64, alpha=0.2, w_neg=1.0, w_bal=0.5):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x n x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute (b x n x n) dot product similarity tensor
        sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # find max for each (n x n) dot product matrix
        max_sim = tf.reduce_max(sims, axis=[-2,-1])
        #min_sim = tf.reduce_min(sims, axis=[-2,-1])
        mean_sim = tf.reduce_mean(sims, axis=[-2,-1])
        
        # HashNet loss, slightly modified
        loss_pos = tf.math.log(1+tf.math.exp(alpha*max_sim)) - max_sim * alpha
        loss_neg = tf.math.log(1+tf.math.exp(alpha*(mean_sim + max_sim)))
        loss_hn = tf.where(y_true == 1, loss_pos, loss_neg * w_neg) # type: ignore

        loss = tf.reduce_sum(loss_hn)

        # hash balance loss: try to push each hash in a sequence to have different values
        # (ideally want all hashes in each half-batch to have different values but that's expensive)

        for ht in (ht1, ht2):
            #(b x n x h) -> (b x n x n)
            loss_bal = tf.matmul(ht, ht, transpose_b=True)
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1]), padding_value=-1) # subtract I
            loss += tf.reduce_mean(loss_bal) * w_bal

        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])

        # get similarity values from 0 to 1
        hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # get max similarity
        max_sim = tf.reduce_max(hash_sims, axis=[-2,-1])

        # predicted positives
        exact_match = tf.where(max_sim == 1.0,
                                max_sim,
                                0 * max_sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        y_true = tf.squeeze(y_true)

        # get similarity values from 0 to 1
        hash_sims = tf.matmul(ht1, ht2, transpose_b=True) / h

        # get max sim
        max_sim = tf.reduce_max(hash_sims, axis=[-2,-1])

        exact_match = tf.where(max_sim == 1.0,
                                max_sim,
                                0 * max_sim)
        
        # get true positives
        tp = exact_match * y_true

        # divide positive sample similarities by # positive samples
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)

        # average hash similarity
        #return tf.reduce_mean(hash_sims)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.sign(y_pred)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

# metrics for single chunk regime
def hash_metric_factory_single(h=64, alpha=0.2, w_neg=1.0, w_bal=0.5):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        
        """
        # HashNet loss, slightly modified
        loss_pos = tf.math.log(1+tf.math.exp(alpha*sim)) - sim * alpha
        loss_neg = tf.math.log(1+tf.math.exp(alpha*sim))
        loss_hn = tf.where(y_true == 1, loss_pos, loss_neg * w_neg) # type: ignore"""

        # simplified loss
        y_true = tf.cast(y_true, float) * 2.0 - 1.0 # type: ignore
        #loss_hn = tf.abs(sim - y_true)

        # new loss: HN w/ pos and neg equally weighted
        loss_hn = tf.math.log(1+tf.math.exp(-y_true * sim * 2))

        loss = tf.reduce_mean(loss_hn)

        # intra-batch loss
        for ht in (ht1, ht2):
            #(b x h) -> (b x b)
            # hash balance loss: try to push each hash in each batch to have different values
            loss_bal = tf.matmul(ht, ht, transpose_b=True) / h
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1])) # subtract I; removed padding_value = -1
            loss += tf.reduce_mean(loss_bal) * w_bal

            # bit uncorrelation loss
            loss_unc = ht * tf.ones_like(ht)
            loss_unc = tf.reduce_mean(loss_unc) ** 2
            loss += loss_unc * w_bal


        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # get predicted positives
        exact_match = tf.where(sim == 1.0,
                                sim,
                                0 * sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        y_true = tf.squeeze(y_true)

        # get similarity values from -1 to 1
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # get predicted positives
        exact_match = tf.where(sim == 1.0,
                                sim,
                                0 * sim)
        
        # get true positives
        tp = exact_match * y_true

        # divide positive sample similarities by # positive samples
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.sign(y_pred)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

# metrics for bucketing
def bucket_metric_factory(n=32, h=64, alpha=0.2, w_neg=1.0, w_bal=0.5, losstype='hinge'):
    # loss given pairwise labels
    def hash_loss(y_true, y_pred):
        # y_true: (b) labels
        # y_pred: list of 2 (b x n x h) hash tensors
        y_true = tf.squeeze(y_true)

        # get hash tensors
        ht1 = y_pred[0]
        ht2 = y_pred[1]

        # compute similarity for each hash pair; this should still work, output (b x n)
        #sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h

        # TODO: roll + tile so we can get a distance matrix between all hashes

        # compute euclidean (l2 norm) distance, get min
        dist = tf.norm(ht1 - ht2, axis=-1) # (b x n)
        min_dist = tf.reduce_min(dist, axis=-1) # (b)

        # compute loss
        z = min_dist - 1.0 # intentionally using 0.5 instead of 1.0 to shift to higher recall
        y_true = tf.cast(y_true, float) * 2.0 - 1.0 # type: ignore
        if losstype == 'hinge':
            loss_hn = tf.maximum(tf.zeros_like(z), 1+y_true*z)
        elif losstype == 'softplus':
            # softplus
            #loss_hn = tf.math.log(1+tf.math.exp(y_true * z))
            # softplus v2 generalized
            alpha = 10.0 # a > 0
            beta = 1.0 # b in [0, 1]
            expa1b = tf.math.exp(alpha*(1-beta)) # this term gets reused a lot
            const = tf.math.log(1+1/expa1b) / alpha + 1/(1+expa1b) # type: ignore
            rescale = 1+1/expa1b
            loss_hn = tf.math.log(1+tf.math.exp(alpha * (y_true * z + beta))) / alpha # type: ignore
            loss_hn -= y_true*z/(1+expa1b) + const
            loss_hn *= rescale
        elif losstype == 'polynomial':
            loss_hn = tf.maximum(tf.zeros_like(z), y_true*(z**3/3 + z) + z**2 + 1/3)
            loss_hn *= 0.25 # type: ignore
        

        loss = loss_hn #tf.reduce_mean(loss_hn)

        # intra-batch loss
        """
        for ht in (ht1, ht2):
            ht = tf.reshape(ht, [-1, h])
            #(b x h) -> (b x b)
            # hash balance loss: try to push each hash in each batch to have different values
            loss_bal = tf.matmul(ht, ht, transpose_b=True) / h
            loss_bal -= tf.linalg.diag(tf.ones(tf.shape(loss_bal)[:-1])) # subtract I; removed padding_value = -1
            loss += tf.reduce_mean(loss_bal) * w_bal

            # bit uncorrelation loss
            loss_unc = ht * tf.ones_like(ht)
            loss_unc = tf.reduce_mean(loss_unc) ** 2
            loss += loss_unc * w_bal"""

        return loss

    # precision: fraction of samples with at least one exact match that are positives
    def hash_precision(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        #ht1 = tf.where(y_pred[0] > 0.5, 1.0, -1.0)
        #ht2 = tf.where(y_pred[1] > 0.5, 1.0, -1.0)

        # compute similarity for each hash pair
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        max_sim = tf.reduce_max(sim, axis=-1)
        

        # get predicted positives
        exact_match = tf.where(max_sim == 1.0,
                               max_sim,
                               0 * max_sim)

        # true positives
        tp = exact_match * tf.squeeze(y_true)

        # mean; add small epsilon to avoid nan
        return tf.reduce_sum(tp) / (tf.reduce_sum(exact_match) + 1e-9)

    # recall: fraction of positive samples with at least one exact match
    def hash_recall(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        #ht1 = tf.where(y_pred[0] > 0.5, 1.0, -1.0)
        #ht2 = tf.where(y_pred[1] > 0.5, 1.0, -1.0)
        y_true = tf.squeeze(y_true)

        # get similarity values from -1 to 1
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        max_sim = tf.reduce_max(sim, axis=-1)

        # get predicted positives
        exact_match = tf.where(max_sim == 1.0,
                               max_sim,
                               0 * max_sim)
        
        # get true positives
        tp = exact_match * y_true

        # compute TP/P
        return tf.reduce_sum(tp) / tf.reduce_sum(y_true)
    
    # fraction of unique hashes; doesn't really need y_true but keras will get mad
    def unique(y_true, y_pred):
        # convert hashes to binary
        hash_bin = tf.where(y_pred > 0.5, 1.0, -1.0)

        # convert to int
        hash_int = tf.reduce_sum(tf.cast(hash_bin, dtype=tf.int64) 
                            * tf.cast(2, dtype=tf.int64) ** tf.range(tf.cast(h, tf.int64)),
                            axis=-1)
        
        # flatten and get unique
        hash_int = tf.reshape(hash_int, [-1])
        hash_unique, _ = tf.unique(hash_int)

        return tf.shape(hash_unique)[0] / tf.shape(hash_int)[0]

    # set names
    hash_precision.__name__ = 'prec'
    hash_recall.__name__ = 'recall'

    return hash_loss, hash_precision, hash_recall, unique

# standalone TNR function
def bucket_TNR_factory(h=64):
    def bucket_TNR(y_true, y_pred):
        # get hash tensors; convert to binary
        ht1 = tf.sign(y_pred[0])
        ht2 = tf.sign(y_pred[1])
        #ht1 = tf.where(y_pred[0] > 0.5, 1.0, -1.0)
        #ht2 = tf.where(y_pred[1] > 0.5, 1.0, -1.0)
        y_true = tf.squeeze(y_true)

        # get similarity values from -1 to 1
        sim = tf.reduce_sum(ht1 * ht2, axis=-1) / h
        max_sim = tf.reduce_max(sim, axis=-1)

        # get predicted negatives
        exact_match = tf.where(max_sim == 1.0,
                               0 * max_sim,
                               max_sim)
        
        # get true negatives
        tn = exact_match * (1.0 - y_true)

        # compute TN/N
        return tf.reduce_sum(tn) / tf.reduce_sum(1.0 - y_true)
    
    bucket_TNR.__name__ = 'TNR'

    return bucket_TNR