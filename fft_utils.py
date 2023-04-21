import numpy as np

# mapping functions

# sets any non-AG character to 1
def char_to_PP(char):
    if char == 'A' or char == 'G':
        return -1
    return 1

# TODO: add more mapping functions

# generic sequence to numerical representation (numpy array) function
def seq_to_num(seq, mapfunc):
    out = np.empty([len(seq)])
    for i in range(0, len(seq)):
        out[i] = mapfunc(seq[i])
    return out

# find mean signal given a list of signals
def cross_correlation_average(signals):
    # Compute the cross-correlation matrix
    num_signals = len(signals)
    cross_corr = np.zeros((num_signals, num_signals), dtype=complex)
    for i in range(num_signals):
        for j in range(num_signals):
            cross_corr[i, j] = np.correlate(signals[i], signals[j])
    
    # Find the reference signal with the highest cross-correlation
    ref_idx = np.argmax(np.sum(cross_corr, axis=1))
    
    # Align the signals with the reference signal
    aligned_signals = []
    for i in range(num_signals):
        if i == ref_idx:
            aligned_signals.append(signals[i])
        else:
            offset = np.argmax(np.correlate(signals[i], signals[ref_idx], mode='same'))
            aligned_signals.append(np.roll(signals[i], -offset))
    
    # Compute the average signal
    avg_signal = np.mean(aligned_signals, axis=0)
    
    return avg_signal

# get sequence from file (ML-DSP)
def seq_from_file(file):
    with open(file, 'r') as f:
        seq = f.read()

    # compute mapping, convert to fft
    # TODO: allow other mappings (once implemented)
    seq = seq_to_num(seq, char_to_PP)
    seq = np.fft.fft(seq, 4096) # pad to max sequence length
    seq = seq / len(seq) # normalize
    seq = np.abs(seq) # find magnitude

    return seq