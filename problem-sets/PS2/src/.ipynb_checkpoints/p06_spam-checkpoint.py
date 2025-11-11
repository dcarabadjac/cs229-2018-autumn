import collections
import re
import numpy as np

from . import util, svm

class NaiveBayes_model():
    """Naive Bayes classifier for binary text classification.

    This class implements a multinomial Naive Bayes model, commonly used for 
    text classification tasks such as spam detection. It learns the conditional 
    probabilities of words given class labels and uses them to predict labels 
    for new messages.

    Attributes:
        theta0 (np.ndarray): Probability of each word given class 0 (e.g., non-spam).
        theta1 (np.ndarray): Probability of each word given class 1 (e.g., spam).
        phi (float): Prior probability of class 1.

    Methods:
        fit(matrix, labels):
            Trains the model by estimating word and class probabilities.
            Uses Laplace smoothing to handle unseen words.

            Args:
                matrix (np.ndarray): Binary or count matrix of shape (m, n), 
                    where m is the number of messages and n is the vocabulary size.
                labels (np.ndarray): Array of shape (m,) containing binary class labels (0 or 1).

            Returns:
                None

        predict(matrix):
            Predicts class labels for new messages based on learned parameters.

            Args:
                matrix (np.ndarray): Binary or count matrix of shape (m, n), 
                    representing messages to classify.

            Returns:
                np.ndarray: Predicted binary class labels (0 or 1) for each input message.
    """
    def __init__(self):
        self.theta0 = None
        self.theta1 = None
        self.phi = None

    def fit(self, matrix, labels):
        m, n = matrix.shape # number of examples vs number of words in dictionary
        assert m == len(labels), 'Mismatch of the number of exampels in matrix and labels'

        mask1 = labels == 1 
        mat_masked = matrix[mask1]
        self.theta1 = (np.sum(mat_masked, axis=0) + 1)/(np.sum(mat_masked) + n)

        mask0 = labels == 0 
        mat_masked = matrix[mask0]
        self.theta0 = (np.sum(mat_masked, axis=0) + 1)/(np.sum(mat_masked) + n)
        
        self.phi = np.mean(labels)

    def predict(self, matrix):
        lnP1 = matrix@np.log(self.theta1) + np.log(self.phi)
        lnP0 = matrix@np.log(self.theta0) + np.log(1-self.phi)

        prediction = lnP1>=lnP0
        prediction = prediction.astype(int)
        return prediction

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

    return message.lower().split()


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
    
    all_words = []
    for message in messages:
        all_words += get_words(message)

    word_counts = collections.Counter(all_words)

    freq_words = [word for word, freq in word_counts.items() if freq >= 5]

    dictionary = {word:i for i, word in enumerate(freq_words)}

    return dictionary

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
    m = len(messages)
    n = len(word_dictionary)
    encoded_messages = np.zeros((m, n))

    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                j = word_dictionary[word]
                encoded_messages[i, j] += 1

    return encoded_messages


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    model = NaiveBayes_model()
    model.fit(matrix, labels)
    return model
    


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    return model.predict(matrix)


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    llr = np.log(model.theta1) - np.log(model.theta0)
    indices_sorted_by_llr = np.argsort(llr)[::-1]
    llr_sorted = llr[indices_sorted_by_llr]

    top_five_indices = indices_sorted_by_llr[:5]

    top_words = []
    for index in top_five_indices:
        top_word = next((k for k, v in dictionary.items() if v==index), None)
        top_words.append(top_word)
    return top_words, llr_sorted

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    best_accuracy = -1
    for radius in radius_to_consider:
        val_predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(val_predictions == val_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_radius = radius
    return optimal_radius

