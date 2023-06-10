import numpy as np

# ML-DSP mapping functions

# mapping 1: Purine-Pyramidine
# sets any non-AG character to 1, can't retrieve original string from ifft
def char_to_PP(char):
    if char == 'A' or char == 'G':
        return -1
    return 1

# mapping 2: Real
# should be possible to (approximately) invert and retrieve the original string
def char_to_real(char):
    if char == 'A':
        return 1.5
    elif char == 'C':
        return 0.5
    elif char == 'G':
        return -0.5
    else:
        return -1.5
    
# mapping 3: Integer
# also should be possible to ifft for original string
def char_to_int(char):
    if char == 'A':
        return 1
    elif char == 'C':
        return 2
    elif char == 'G':
        return 3
    else:
        return 4

# generic sequence to numerical representation (numpy array) function
def seq_to_num(seq, mapfunc):
    out = np.empty([len(seq)])
    for i in range(0, len(seq)):
        out[i] = mapfunc(seq[i])
    return out

# MAFFT-style FFT
# sequence to one-hot encoding
def seq_to_oh(seq):
    out = np.empty([len(seq), 4], dtype=bool)
    for i in range(0, len(seq)):
        if seq[i] == 'A':
            out[i] = [1, 0, 0, 0]
        elif seq[i] == 'C':
            out[i] = [0, 1, 0, 0]
        elif seq[i] == 'G':
            out[i] = [0, 0, 1, 0]
        else:
            out[i] = [0, 0, 0, 1]
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

# get sequence from file
def seq_from_file(file, method='ML-DSP'):
    with open(file, 'r') as f:
        seq = f.read()

    if method == 'ML-DSP':
        # compute mapping, convert to FFT; default mapping = char_to_PP
        seq = seq_to_num(seq, char_to_PP)
        seq = np.fft.fft(seq, 4096) # pad to max sequence length
    elif method == 'ML-DSP-REAL':
        seq = seq_to_num(seq, char_to_real)
        seq = np.fft.fft(seq, 4096)
    elif method == 'ML-DSP-INT':
        seq = seq_to_num(seq, char_to_int)
        seq = np.fft.fft(seq, 4096)
    elif method == 'MAFFT':
        # convert to one-hot encoding, convert each row to FFT, transpose for easier correlation
        # when handling, check len(seq.shape); if len == 1, proceed as usual, otherwise correlate 4x + sum
        seq = seq_to_oh(seq)
        seq = np.fft.fft(seq, 4096, axis=0)
        seq = np.transpose(seq)
    elif method == 'MAFFT-COMPRESSED':
        # compute sum of all 4 one-hot FFTs
        seq = seq_to_oh(seq)
        seq = np.fft.fft(seq, 4096, axis=0)
        seq /= len(seq) # normalize
        seq = np.abs(seq) # find magnitude
        seq = np.sum(seq, axis=1)
        return seq
    else:
        raise ValueError("invalid method string; options: ML-DSP, ML-DSP-REAL, ML-DSP-INT, MAFFT")

    seq /= len(seq) # normalize
    seq = np.abs(seq) # find magnitude

    return seq
