import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # # Normalize the string
    # message.rstrip()
    # message.lower()
    # wordlist = []
    # msg_len = len(message)
    # flag = 0
    # for i in range(msg_len):
    #     if message[i] == ' ':
    #         wordlist.append(message[flag:i])
    #         flag = i+1
    # # wordlist.append(message[flag:])
    # msg_remain = message[flag:].strip()
    # # while not msg_remain.isalpha():
    # #     msg_remain = msg_remain[:-1]
    # #     # msg_remain = msg_remain.rstrip(msg_remain[-1])
    # wordlist.append(msg_remain)
    word_list = message.lower().split()
    if word_list[-1].isalpha():
        pass
    else:
        word_list[-1].strip(',./?<>{}[]\\|\/()_+-=!@#$%^&*')
    return word_list
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dictionary = dict()
    index = 0
    for message in messages:
        for word in get_words(message):
            if word not in dictionary:
                dictionary[word] = index
                index += 1
    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    word_frequency = np.array(word_dictionary)
    # # Initialize word frequency
    # for a_word in word_frequency:
    #     word_frequency[a_word] = 0
    # Record occurrence of words in messages
    for message in messages:
        for word in message:
            if word in word_dictionary:
                word_frequency[word] += 1
    return word_frequency
    # *** END CODE HERE ***

SMS = ["What you doing?how are you?"
       "Ok lar... Joking wif u oni..."
       "dun say so early hor... U c already then say..."
       "MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*"
       "Siva is in hostel aha:-."
       "Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who"]

print(get_words("MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*"))
