"""Things that are handy to have around"""
import numpy as np

def one_big_sequence(data, highest, lowest):
    """Turns a sequence x time x notes list of lists of lists into
    a big numpy array of sequence x notes by sticking them all together.
    Just uses a 1.0 for all of the notes that are present and a special
    symbol for end of sequence.

    Needs to be told the highest and lowest notes to account for as this may
    not be the same across all splits of the data. Makes an end of sequence
    symbol equivalent to highest+1

    Args:
        data: nested lists of ints, representing the data per tune per
            timestep (with multiple notes potentially active at each step)
        highest: the highest note
        lowest: the lowest note.
    """
    num_features = (highest - lowest) + 1
    eos = highest + 1
    big_sequence = [item for sequence in data for item in (sequence + [[eos]])]
    np_data = np.zeros((len(big_sequence), num_features), dtype=np.float32)

    # now fill in the ones
    for i, notes in enumerate(big_sequence):
        np_data[i, tuple(note-lowest-1 for note in notes)] = 1.0

    return np_data
