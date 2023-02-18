import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import itertools

# Natural Language module
from nltk.tokenize import word_tokenize

import string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

# Regex
import re

# Matplotlib, graphical visualization
import matplotlib.pyplot as plt

# SECTION 1 - Text Pretreatment < Dependant on data source
def text_pretreatment(sentences):
    print(f"~~~ Text pretreatment underway. ~~~")

    # Remove \n
    processed_sentences = []

    for sentence in sentences:
        processed_sentences.append(sentence.replace('\n', ''))

    print(f"processed_sentences > \n{processed_sentences}")
    return processed_sentences


# SECTION 2 - Text Vectorization  < Standard once data is pretreated
def word_vectorization(text_ds, sequence_length, vocab_size=None):
    print(f"~~~ Text vectorization underway. ~~~")

    vectorization_layer = TextVectorization(
        standardize="lower_and_strip_punctuation",
        output_mode='int',
        output_sequence_length=sequence_length)

    # Original
    # vectorization_layer = TextVectorization(
    #     standardize="lower_and_strip_punctuation",
    #     max_tokens=vocab_size,
    #     output_mode='int',
    #     output_sequence_length=sequence_length)


    # Batch selects number of samples to adapt the layer. Generates vectors for each word
    vectorize_results = vectorization_layer.adapt(text_ds.batch(1024))  # Adapt Vectorizations to 1024 samples

    # Save the created vocabulary to create decoder from word vectors
    inverse_vocab = vectorization_layer.get_vocabulary()
    print(f"Vectorize Layers Vocabulary > \n{inverse_vocab}")

    ls_vectorized_sentences = []
    # Visualization of Vectorized Sentence. Array set to 100 word vectors.
    for iter, processed_sentence in enumerate(processed_sentences):
        ls_vectorized_sentences.append(vectorization_layer(processed_sentence))
        print(f"Vectorized sentence {iter} >\n{ls_vectorized_sentences[iter]}")  # Test run, not actual

    return ls_vectorized_sentences,inverse_vocab


# SECTION 3 Null - Generate Skipgrams (Prototype)
def singular_skipgram_generation(ls_vectorized_sentences, inverse_vocab, sentence_index=0,
                                 num_ns=4, window_size=2, SEED=42):
    print(f"~~~ Skipgram generation underway. ~~~")

    # Sampling Table generation for random sampling - to randomly sample from vocabulary to get negative skipgrams
    # sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
    # print(f"sampling_table > {sampling_table}")

    # Skipgram for one sentence
    vocab_size = len(inverse_vocab)

    # Positive Skipgrams for first Sentence
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
        ls_vectorized_sentences[sentence_index],
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples=0)

    print(f"len(positive_skip_grams) > {len(positive_skip_grams)}")

    for target, context in positive_skip_grams:
        print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

    # Generate 4 Negative Skipgrams from one Target Word
    target_word, context_word = positive_skip_grams[0]
    print(f"target_word, context_word > {target_word}, {context_word}")

    # Convert context word into a tensor to be consistent in format and allow contacenation for Skipgram embedding
    context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

    print(f"context_class > {context_class}")

    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=context_class,  # class that should be sampled as 'positive'
        num_true=1,  # each positive skip-gram has 1 positive context class
        num_sampled=num_ns,  # number of negative context words to sample
        unique=True,  # all the negative samples should be unique
        range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
        seed=SEED,  # seed for reproducibility
        name="negative_sampling"  # name of this operation
    )
    print(f"negative_sampling_candidates > {negative_sampling_candidates}")
    print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

    # Merge Positive and Negative Skipgram Templates for Processing
    # Reduce a dimension so you can use concatenation (in the next step).
    squeezed_context_class = tf.squeeze(context_class, 1)

    # Concatenate a positive context word with negative sampled words.
    context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)

    # Label the first context word as `1` (positive) followed by `num_ns` `0`s (negative).
    label = tf.constant([1] + [0] * num_ns, dtype="int64")
    target = target_word

    print(f"target_index    : {target}")
    print(f"target_word     : {inverse_vocab[target_word]}")
    print(f"context_indices : {context}")
    print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
    print(f"label           : {label}")

    print("target  :", target)
    print("context :", context)
    print("label   :", label)

# SECTION 3  - Generate Skipgrams
def generate_skipgrams(ls_vectorized_sentences, inverse_vocab, window_size=2, num_ns=4, SEED=42):
    print(f"~~~ Generate skipgrams from vectorized sentences. ~~~")
    # Skipgram for one sentence
    vocab_size = len(inverse_vocab)

    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Bug Here. It seems to reduce the probability of sampling significantly. Probably not usable for small data sets
    # sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    # print(sampling_table)

    for vectorized_sentence in ls_vectorized_sentences:
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            vectorized_sentence,
            vocabulary_size=vocab_size,
            # sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # print(f"positive_skip_grams > {positive_skip_grams}")

        for target_word, context_word in positive_skip_grams:
            num_ns = 4

            # Convert context word into a tensor to be consistent in format and allow contacenation for
            # Skipgram embedding.
            context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,  # class that should be sampled as 'positive'
                num_true=1,  # each positive skip-gram has 1 positive context class
                num_sampled=num_ns,  # number of negative context words to sample
                unique=True,  # all the negative samples should be unique
                range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
                seed=SEED,  # seed for reproducibility
                name="negative_sampling"  # name of this operation
            )

            # Merge Positive and Negative Skipgram Templates for Processing
            # Reduce a dimension so you can use concatenation (in the next step).
            squeezed_context_class = tf.squeeze(context_class, 1)

            # Concatenate a positive context word with negative sampled words.
            context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)

            # Label the first context word as `1` (positive) followed by `num_ns` `0`s (negative).
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


# SECTION 4- Word2Vec Skipgram Word Embedding
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns+1)

    def call(self, pair):
        target, context = pair

        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)

        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        # Apply dot operation to combine word embedding to context embedding
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots


if __name__ == '__main__':
    # SECTION 0 - Text Data Ingress < Dependant on data source. Current source is direct from .txt file
    with open('textdata.txt') as f:
        sentences = f.readlines()

    processed_sentences = text_pretreatment(sentences)

    # Convert processed_sentences into Dataset for Vectorization Layer - Part of Data Ingress process for Tensorflow
    # REFERENCE: https://www.tensorflow.org/tutorials/load_data/text#example_1_predict_the_tag_for_a_stack_overflow_question
    # REFERENCE: https://www.tensorflow.org/guide/data
    # REFERENCE: https://www.tensorflow.org/text
    # .from_tensor_slices converts raw data into tf.Tensor required to input into Tensorflow methods
    text_ds = tf.data.Dataset.from_tensor_slices(processed_sentences)  # To separate out

    vectorized_sentences, inverse_vocab = word_vectorization(text_ds, sequence_length=25)

    # Prototype function to build up the process of generating Word2Vec Skipgram Embedding
    # singular_skipgram_generation(vectorized_sentences, inverse_vocab)

    # Full Skipgram Generation from sentences
    targets, contexts, labels = generate_skipgrams(vectorized_sentences, inverse_vocab)
    # BUG Here: No targets, contexts, and labels generated

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)


    print(f"targets > \n{targets}\n")
    print(f"contexts > \n{contexts}\n")
    print(f"labels > \n{labels}\n")

    print(f"targets.shape > {targets.shape}")
    print(f"contexts.shape > {contexts.shape}")
    print(f"labels.shape > {labels.shape}")

    # SECTION 4 Execution
    # BATCH_SIZE = 128  # Too Large Batch Size caused an error upon .fit()
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    # Convert targets, contexts and labels into a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

    # TODO Check Batch Size and how it works
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(f"dataset > {dataset}")

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)  # Improve performance for large datasets

    embedding_dim = 128
    vocab_size = len(inverse_vocab)
    num_ns = 4

    word2vec = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, num_ns=num_ns)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])  # run_eagerly=True,

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")  # Optional

    # word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
    word2vec.fit(dataset, epochs=20)

    # Post Processing
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    print(f"Word2Vec weights >\n{weights}")
