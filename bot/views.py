# Importing the required modules
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Activation, Dropout
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
import nltk
import numpy as np
import pickle
import json
import random
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import logging
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Model
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger('django')
le = LabelEncoder()

word_lemmatizer = WordNetLemmatizer()
# Creating a WordNetLemmatizer object to lemmatize words
intent_data = open('bot/static/Intent - loyalist.json').read()

# Loading the JSON file containing the chatbot intents
intents = json.loads(intent_data)
# nltk.download('punkt')

data1 = intents

#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in data1['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
data = pd.DataFrame({"inputs":inputs,
                        "tags":tags})
# data = data.sample(frac=1)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])

train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding

x_train = pad_sequences(train)

#encoding the outputs
y_train = le.fit_transform(data['tags'])

# Create your views here.

def train_model(request):
    
    input_shape = x_train.shape[1]
    print(input_shape)

    vocabulary = len(tokenizer.word_index)
    print("number of unique words : ",vocabulary)
    output_length = le.classes_.shape[0]
    print("output length: ",output_length)

    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary+1,10)(i)
    # x = Dense(128, input_shape=(input_shape,), activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(64, input_shape=(input_shape,), activation='relu')(x)
    # # x = Dropout(0.2)(x)
    # x = Dense(32, input_shape=(input_shape,), activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = LSTM(30,return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length,activation="softmax")(x)
    model  = Model(i,x)

    model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

    train = model.fit(x_train,y_train,epochs=50)
    model.save("bot/mlmodels/chatbot_model.h5", model)
    return HttpResponse('training completed')

def generate_response(request, user_input="hello"):
    model = load_model('bot/mlmodels/chatbot_model.h5')
    
    train = tokenizer.texts_to_sequences(data['inputs'])
    #apply padding
    
    x_train = pad_sequences(train)

    input_shape = x_train.shape[1]

    texts_p = []
    logger.info(f'request info: {request}')
    if request is not None:
        user_input = request

    prediction_input = user_input
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    #getting output from model
    
    output = model.predict(prediction_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    print("Loyalist Bot : ",random.choice(responses[response_tag]))
    chat_resp = random.choice(responses[response_tag])
    
    chat_resp_json = {'response': chat_resp}
    return JsonResponse(chat_resp_json)

def generate_response_old(request, user_input="hello"):

    logger.info(f'request info: {request}')
    if request is not None:
        user_input = request
    # loading the files saved previously
    words = pickle.load(open('bot/static/words.pkl', 'rb'))
    word_classes = pickle.load(open('bot/static/classes.pkl', 'rb'))

    logger.info('some sample text')
    ints = class_prediction(input_words=user_input,
                            words=words, word_classes=word_classes)
    chat_resp = chat_response(ints, intents)
    logger.info(chat_resp)
    chat_resp_json = {'response': chat_resp}

    return JsonResponse(chat_resp_json)

def train_model_old(request):

    # Creating empty lists to store the lemmatized words, their class, and the overall word collection
    lem_words = []
    word_class = []
    word_collection = []

    # Creating a list of regular expressions to ignore
    ignore_regex = ["?", "!", ".", ","]

    logger.info(intents['intents'])
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # separating words from patterns
            word_list = nltk.word_tokenize(pattern)
            lem_words.extend(word_list)  # and adding them to words list

            # associating patterns with respective tags
            word_collection.append(((word_list), intent['tag']))

            # appending the tags to the class list
            if intent['tag'] not in word_class:
                word_class.append(intent['tag'])

    # storing the root words or lemma
    lem_words = [word_lemmatizer.lemmatize(word)
                 for word in lem_words if word not in ignore_regex]
    lem_words = sorted(set(lem_words))

    # Saving the list of root words to a binary file
    pickle.dump(lem_words, open('bot/static/words.pkl', 'wb'))

    # Saving the list of word collections and their respective tags to a binary file
    pickle.dump(word_class, open('bot/static/classes.pkl', 'wb'))

    logger.info(lem_words)
    logger.info(word_class)
    logger.info(word_collection)

    # Preparing the chatbot training data with numerical values for a neural network
    training_data = []  # list to store the prepared data
    # empty list to store the labels in a binary format
    container = [0]*len(word_class)

    for collection in word_collection:
        bag_of_words = []
        w_pattern = collection[0]
        w_pattern = [word_lemmatizer.lemmatize(
            word.lower()) for word in w_pattern]

        for w in lem_words:
            bag_of_words.append(
                1) if w in w_pattern else bag_of_words.append(0)

        # making a copy of the output_empty
        output_data = list(container)
        output_data[word_class.index(collection[1])] = 1
        training_data.append([bag_of_words, output_data])

    random.shuffle(training_data)  # shuffle the prepared data
    # convert the prepared data to a numpy array
    t_data = np.array(training_data)
    # return the features and labels as lists
    X, Y = list(t_data[:, 0]), list(t_data[:, 1])

    logger.info(X)
    logger.info(Y)

    # Convert feature and label lists to numpy arrays
    train_x, train_y = np.array(X), np.array(Y)

    # Determine the number of input and output nodes for the neural network
    input_shape = len(train_x[0])   # the number of features in the dataset
    output_shape = len(train_y[0])  # the number of classes in the dataset

    # Print the shapes of the numpy arrays
    logger.info(train_x.shape)  # prints the shape of the train_x array
    logger.info(train_y.shape)  # prints the shape of the train_y array

    model = Sequential()

    # ======================================================
    # # Adding a densely connected layer with 128 nodes, with ReLU activation function and input shape equal to input
    # model.add(Dense(1024, input_shape=(input_shape,), activation='relu'))
    # model.add(Dropout(0.25))  # Adding a dropout layer to prevent overfitting
    # # Adding a densely connected layer with 64 nodes and ReLU activation function
    # model.add(Dense(512, activation='relu'))
    # # Adding another dropout layer to prevent overfitting
    # model.add(Dropout(0.25))
    # # Adding a densely connected layer with 64 nodes and ReLU activation function
    # model.add(Dense(256, activation='relu'))
    # # Adding another dropout layer to prevent overfitting
    # model.add(Dropout(0.5))
    # # Adding the final output layer with output shape equal to the number of classes and softmax activation function
    # model.add(Dense(output_shape, activation='softmax'))
    # ======================================================
    # Adding a densely connected layer with 128 nodes, with ReLU activation function and input shape equal to input
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    # =====================================================

    # Compiling the model with categorical cross-entropy as loss function, Adam optimizer
    sgd = SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    logger.info(model.summary())

    # Training the model with train_x(features) and train_y(labels), with 100 epochs and a batch size of 10, with verbose set to 1
    trained_model = model.fit(train_x, train_y, epochs=100, batch_size=5)

    # Saving the trained model to a file named chatbot_adam_model.h5
    model.save("bot/mlmodels/chatbot_sgd_model.h5", trained_model)

    # build RNN Model with tensorflow

    model1 = Sequential([
        Dense(512, input_shape=(input_shape,), activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.01)
    model1.compile(optimizer=optimizer,
                   loss='categorical_crossentropy', metrics=['accuracy'])
    model1.summary()

    # early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    # train the model
    tr = model.fit(train_x, train_y, epochs=100, batch_size=5)

    model.save("bot/mlmodels/chatbot_adam_model.h5", tr)

    return HttpResponse('training completed')


def process_input(input_words):
    logger.info(f'trying to process input: {input_words}')
    tokenized_words = nltk.word_tokenize(input_words)
    tokenized_words = [word_lemmatizer.lemmatize(
        word) for word in tokenized_words]
    print(f'tokenized words: {tokenized_words}')

    return tokenized_words


def bag_of_words(input_words, words):

    # separate out words from the input sentence
    input_words = process_input(input_words)
    word_bag = [0]*len(words)
    for w in input_words:
        for i, word in enumerate(words):

            # check whether the word
            # is present in the input as well
            if word == w:

                # as the list of words
                # created earlier.
                word_bag[i] = 1

    # return a numpy array
    return np.array(word_bag)


def class_prediction(input_words, words, word_classes):
    logger.info('inside class_prediction function')
    model = load_model('bot/mlmodels/chatbot_model.h5')
    # model = load_model('bot/mlmodels/chatbot_adam_model.h5')

    feature_words = bag_of_words(input_words, words)
    resp = model.predict(np.array([feature_words]))[0]
    ERROR_THRESHOLD = 0.25
    response = [[i, r] for i, r in enumerate(resp) if r > ERROR_THRESHOLD]
    response.sort(key=lambda x: x[1], reverse=True)
    response_list = []

    for r in response:
        response_list.append(
            {'intent': word_classes[r[0]], 'probability': str(r[1])})

    logger.info(f'class pred: {response_list}')
    return response_list


def chat_response(w_list, intents_file):
    logger.info(f'wlist: {w_list}')
    tag = w_list[0]['intent']
    list_of_intents = intents_file['intents']
    # print(f'tag: {tag}, {tag[1]}')
    # print(f'list of intents: {list_of_intents}')
    response = ""
    for intent in list_of_intents:
        # print(f'poss intent tag: {intent["tag"]}')
        if intent['tag'] == tag:
            # prints a random response
            print(f'matched intent tag: {intent["tag"]}')
            response = random.choice(intent['responses'])
            break
    logger.info(f'chat response: {response}')
    return response



