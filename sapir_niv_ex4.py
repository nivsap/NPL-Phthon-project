import os
import random
import re
import string
from collections import Counter
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
import matplotlib.pyplot as plt

# The Corpus class from previous exercises:
class Token:

    def __init__(self, t, c, p, h, nam, o, v):
        self.type = t  # word or char
        self.c5 = c  # c5 tag
        self.pos = p  # pos = part of speech
        self.hw = h  # head word
        self.name = nam  # usually empty in our corpus
        self.occurs = o  # numbers of occurs to each word
        self.value = v  # value of token
        return


class Sentence:

    def __init__(self, tokens, t):
        self.tokens = tokens  # arr of tokens
        self.title = t
        return


class Corpus:
    def __init__(self):
        self.sentences = []  # arr of sentences
        self.sentences_lengths = []
        self.chunks = []
        self.gender_of_chunks = []

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        tree = ET.parse(file_name)
        authors_arr = []
        # iterate over sentence of the xml files
        for sentence in tree.iter(tag='s'):
            tokens_arr = []
            for word in sentence:
                if word.tag in ('w', 'c') and isinstance(word.text, str):
                    att = word.attrib
                    new_token = Token(
                        t=word.tag,
                        c=att['c5'],
                        h=att.get('hw'),
                        p=att.get('pos'),
                        nam=" ",
                        o=" ",
                        v=word.text
                    )
                    tokens_arr.append(new_token)
            tokens_arr = np.array(tokens_arr)
            new_sentence = Sentence(tokens_arr, "")
            self.sentences.append(new_sentence)
        return

    def get_tokens(self):
        tokens_list = []
        for sen in self.sentences:
            for token in sen.tokens:
                tokens_list.append(token.value)
        return tokens_list

def tokenize_words(sentences):
    """
    tokenize the words in the sentences.
    :param sentences:
    :return: list of sentences that includes tokens.
    """
    # sentence_temp = tokenize_special_characters(sentences)
    tokens = []
    # if sentence_temp.startswith(" "):
    #     sentence_temp = sentence_temp[1:]
    words = sentences.split(" ")
    # create tokens
    for word in words:
        if (word in string.punctuation):
             word_or_char = False
        else:
            word_or_char = True
        new_token = Token('word' if word_or_char else 'char', "", "", "", "", "", word)
        tokens.append(new_token)
    return list(filter(lambda token: token != '', sentences.split(' ')))


def tokenize_special_characters(sentence):
    """
    Insert whitespace before and after: '(' , ')' , ';' , '<' '>'
     '!' , '?' , '+' , '=' , '"' , '%' , '#' , '^' , '{' , '}' , '[' ']' ,
      ',' , '.' , ':' , ';' , '/' , '\'
    :param sentence: sentence.
    :return: sentence
    """
    regex = re.compile(r'(/|<|>|\(|\)|;|=|\{|\[|\]|\+|\^|\%|\}|\#|"|\|)')
    sentence = regex.sub(r' \1 ', sentence)
    regex = re.compile(r'(?<!\d)' + r'(\.|\\|/|:)' + r'(?!\d|\s)')
    sentence = regex.sub(r' \1 ', sentence)
    regex = re.compile(r'(?<!\d|\s)' + r'(\.|:|\\|/)' + r'(?!\d)')
    sentence = regex.sub(r' \1 ', sentence)
    sentence = re.sub(r'[,]+(?![0-9])', r' , ', sentence)
    sentence = re.sub("  +", " ", sentence)
    if ("i . e ." in sentence):
        sentence = re.sub("i . e .", lambda params: params[0].replace(' ', ""), sentence)
    if ("e . g ." in sentence):
        sentence = re.sub("e . g .", lambda params: params[0].replace(' ', ""), sentence)
    return sentence


def save_glove_text_file_once(file_path):

    glove_input_dir = 'kv_files'
    word2vec_output_dir = 'word2vec_files'
    key_vectors_dir = 'key_vectors'

    for i, file in enumerate(os.listdir(glove_input_dir)):
        file_num = file.split('.')[2]
        downloaded_text_filename = os.path.join(os.getcwd(), glove_input_dir, file)
        full_path_vector_filename = os.path.join(os.getcwd(), word2vec_output_dir, file_num)
        glove2word2vec(downloaded_text_filename, full_path_vector_filename)
        pre_trained_model = KeyedVectors.load_word2vec_format(full_path_vector_filename, binary=False)
        pre_trained_model.save(os.path.join(os.getcwd(), key_vectors_dir, f'{file_num}.kv'))
## Do the following only once!

## Save the GloVe text file to a word2vec file for your use:
# glove2word2vec(<downloaded_text_filename>, <full_path_vector_filename>')
## Load the file as KeyVectors:
# pre_trained_model = KeyedVectors.load_word2vec_format(<full_path_vector_filename.kv>, binary=False)
## Save the key vectors for your use:
# pre_trained_model.save(<full_path_keyvector_filename.kv>)

## Now, when handing the project, the KeyVector filename will be given as an argument.
## You can load it as follwing:
# pre_trained_model = KeyedVectors.load(<full_path_keyvector_filename>.kv, mmap='r')


def task_a(model_kv):

    #1

    output_str = 'Word Pairs and Distances:\n'
    distance = model_kv.similarity("fast", "rapid")
    output_str += "1. " + "fast - rapid : " +str(distance) + "\n"
    distance = model_kv.similarity("fast", "slow")
    output_str += "2. " + "fast - slow : " + str(distance) + "\n"
    distance = model_kv.similarity("tale", "story")
    output_str += "3. " + "tale - story : " + str(distance) + "\n"
    distance = model_kv.similarity("real", "actual")
    output_str += "4. " + "real - actual : " + str(distance) + "\n"
    distance = model_kv.similarity("short", "long")
    output_str += "5. " + "short - long : " + str(distance) + "\n"
    distance = model_kv.similarity("silent", "quiet")
    output_str += "6. " + "silent - quiet : " + str(distance) + "\n"
    distance = model_kv.similarity("help", "support")
    output_str += "7. " + "help - support : " + str(distance) + "\n"
    distance = model_kv.similarity("reply", "answer")
    output_str += "8. " + "reply - answer : " + str(distance) + "\n"
    distance = model_kv.similarity("help", "support")
    output_str += "9. " + "help - support : " + str(distance) + "\n"
    distance = model_kv.similarity("pear", "apple")
    output_str += "10. " + "pear - apple : " + str(distance) + "\n"

    #2
    distances_str = '\nDistances:\n'
    output_str += "\nAnalogies:\n"
    output_str += "1. man : woman , boy : girl\n"
    output_str += "2. shoe : foot , hat : head\n"
    output_str += "3. whole : part , school : classroom\n"
    output_str += "4. effect : cause , flood : rain\n"
    output_str += "5. cause : effect , practice : improve\n"

    output_str += "\nMost Similar:\n"
    return_word = model_kv.most_similar(positive=["woman", "boy"], negative=["man"])[0][0]
    output_str += "1. woman + boy - man = " + str(return_word) + "\n"
    distance = model_kv.similarity(return_word, "girl")
    distances_str += "1. girl - " + str(return_word) + " : " + str(distance) + "\n"
    return_word = model_kv.most_similar(positive=["foot", "hat"], negative=["shoe"])[0][0]
    output_str += "2. foot + hat - shoe = " + str(return_word) + "\n"
    distance = model_kv.similarity(return_word, "head")
    distances_str += "2. head - " + str(return_word) + " : " + str(distance) + "\n"
    return_word = model_kv.most_similar(positive=["part", "school"], negative=["whole"])[0][0]
    output_str += "3. part + school - whole = " + str(return_word) + "\n"
    distance = model_kv.similarity(return_word, "classroom")
    distances_str += "3. classroom - " + str(return_word) + " : " + str(distance) + "\n"
    return_word = model_kv.most_similar(positive=["cause", "flood"], negative=["effect"])[0][0]
    output_str += "4. cause + flood - effect = " + str(return_word) + "\n"
    distance = model_kv.similarity(return_word, "rain")
    distances_str += "4. rain - " + str(return_word) + " : " + str(distance) + "\n"
    return_word = model_kv.most_similar(positive=["effect", "practice"], negative=["cause"])[0][0]
    output_str += "5. effect + practice - cause = " + str(return_word) + "\n"
    distance = model_kv.similarity(return_word, "improve")
    distances_str += "1. improve - " + str(return_word) + " : " + str(distance) + "\n"

    output_str += distances_str

    return output_str


class NGramModel:    ######################
    def __init__(self, n_gram, corpus):
        self.corpus = corpus
        self.n_gram = n_gram
        self.size_of_vocabulary = []

        tokens = self.corpus.get_tokens()
        token_dic = dict()
        for token in tokens:
            if (token_dic.get(token) == None):
                token_dic[token] = 1
            else:
                token_dic[token] += 1
        self.size_of_vocabulary.append(len(token_dic))
        self.num_of_words = len(tokens)
        uniq_words = self.create_kGrams(tokens)
        num_of_uniq_words = []
        for group in uniq_words:
            num_of_uniq_words.append(dict(Counter(group)))
        self.num_of_tokens_whole_corpus = num_of_uniq_words

    def create_kGrams(self, tokens_list):
        uniq_words = [tokens_list]
        for k in range(2, n_gram + 1):
            k_groups = []
            for i in range(len(tokens_list) - k + 1):
                k_groups.append(' '.join(tokens_list[i:i + k]))
            uniq_words.append(k_groups)
            temp_dict = dict()
            for group in k_groups:
                if temp_dict.get(group) == None:
                    temp_dict[group] = 1
                else:
                    temp_dict[group] += 1
            self.size_of_vocabulary.append(len(temp_dict))
        return uniq_words



def task_b(lyrics_file, model: KeyedVectors, n_gram_model):
    max_counter = 0
    words_to_change = [
        'said', 'doing', 'you', 'you', 'say',
        'sipping', 'to', 'clean', 'shaved', 'should',
        'wing', 'this', 'playing', 'from', 'arms',
        'leave', 'door', 'open', 'the', 'feel', 'baby',
        'me', 'tight', 'bite', 'haze', 'got', 'me',
        'we', 'kissing', 'petals', 'jump', 'games',
        'straight', 'arms', 'door', 'leave', 'leave',
        'door', 'feel', 'me', 'come', 'baby',
        'see', 'give', 'ah', 'door', 'leave',
        'open', 'way', 'tonight', 'coming', 'me',
        'tell', 'woo', 'woo', 'la', 'coming', 'waiting',
        'over', 'waiting', 'tell', 'for', 'you',
        'la'
    ]
    output = '=== New Hit ===\n'
    with open(lyrics_file) as lyrics_song:
        for line, word_to_change in zip(lyrics_song, words_to_change):
            line = line.replace('\n', '')
            tokens = tokenize_sentence(line)  ######
            places_of_words = []
            for i in range(len(tokens)):
                if tokens[i].lower() == word_to_change.lower():
                    places_of_words.append(i)
            replacements_arr = model.most_similar(word_to_change)
            for index in places_of_words:
                our_tokens = define_limits(places_of_words, len(tokens), tokens)
                our_match = replacements_arr[0][0]
                for replacement, _ in replacements_arr:
                    search_text = ' '.join(our_tokens).replace(word_to_change, replacement)
                    counter = n_gram_model.num_of_tokens_whole_corpus[2].get(search_text, 0)
                    if counter > max_counter:
                        max_counter = counter
                        our_match = replacement
                if max_counter == 0:
                    for replacement, _ in replacements_arr:
                        sum1 = 0
                        sum2 = 0
                    if index != 0:
                        part1 = ' '.join(tokens[index - 1: index + 1]).replace(word_to_change,  replacement)
                        sum1 = n_gram_model.num_of_tokens_whole_corpus[1].get(part1, 0)
                    counter = sum1 + sum2
                    if counter > max_counter:
                        max_counter = counter
                        our_match = replacement
            word = re.compile(word_to_change, re.IGNORECASE)
            line_of_song = word.sub(our_match, line)
            output += line_of_song + '\n'

    return output

class Tweet:
    def __init__(self, tokens, category, index):
        self.tokens = tokens
        self.index = index
        self.category = category


def define_limits(places_of_words, tokens_len, tokens):
    for i in places_of_words:
        if i != 0 and i != tokens_len - 1:
            start_index = i - 1
            end_index = i + 1
        elif i == 0:
            start_index = i
            end_index = i + 2
        else:
            start_index = i - 2
            end_index = i
    toReturn = []
    for i in range(tokens_len):
        if i >= start_index and i <= end_index + 1:
            toReturn.append(tokens[i])
    return toReturn


def tokenize_sentence(sentence):
    sentence = sentence.replace('…', '...').replace('’', "'").replace('“', '"')
    for sign in r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""":
        sentence = sentence.replace(sign, ' ' + sign + ' ')
    sentence = sentence.replace('.  .  .', '...').replace('\'', ' \'').replace('n \'t', ' n\'t')
    tokens = list(filter(lambda token: token != '', sentence.split(' ')))
    return tokens


def define_to_tweets(our_category, tweets_dict):
    index = 0
    category = None
    tweets = []
    with open(tweets_dict, encoding='utf8') as file:
        for sen in file:
            temp_sen = sen.replace('\n', '').strip()
            if temp_sen in our_category:
                category = temp_sen
                index = 1
            elif temp_sen != '':
                sen_to_token = tokenize_sentence(temp_sen)
                index += 1
                newTweet = Tweet(sen_to_token, category, index)
                tweets.append(newTweet)
    return tweets


def create_vector(weight, model, tweet, all_scores):

    weight_vector = []
    v = []
    for token in tweet.tokens:
        if weight == 2:
            weight = random.randint(1, 10)
        if weight == 3:
            if '== Covid ==' in tweet.category:
                weight = all_scores.get(token) + 100
            elif '== Olympics ==' in tweet.category:
                weight = all_scores.get(token) - 100
            else:
                weight = all_scores.get(token) - 50
        weight_vector.append(np.full(model.vector_size, weight))
        if token.lower() in model:
            v.append(model[token.lower()])
        else:
            v.append(np.ones(model.vector_size))
    weight_vector = np.array(weight_vector)
    v = np.array(v)
    new_vec = np.zeros(model.vector_size)
    for i in range(len(tweet.tokens)):
        new_vec += np.multiply(weight_vector[i], v[i])
    new_vec = new_vec / len(tweet.tokens)
    return new_vec


def task_c(tweets_file, model):
    our_category = [
        '== Covid ==',
        '== Olympics ==',
        '== Pets =='
    ]
    all_my_vectors = []
    tweets = define_to_tweets(our_category, tweets_file)
    all_scores = dict()
    for tweet in tweets:
        for token in tweet.tokens:
            if (all_scores.get(token) == None):
                all_scores[token] = 1
            else:
                all_scores[token] += 1
    pca = PCA(n_components=2)
    weight = 1
    for tweet in tweets:
        all_my_vectors.append(create_vector(weight, model, tweet, all_scores))
    pca.fit(all_my_vectors)
    transformed = pca.transform(all_my_vectors)
    index1 = 0
    plt.title(f'Arithmetic - Niv Sapir')
    for (x_point, y_point) in transformed:
        plt.scatter(x_point, y_point, s=8)
        plt.text(x_point, y_point, f'{tweets[index1].category}-{tweets[index1].index}', fontsize=7)
        index1 += 1
    plt.show()

    all_my_vectors = []
    for tweet in tweets:
        all_my_vectors.append(create_vector(2, model, tweet, all_scores))
    pca.fit(all_my_vectors)
    transformed = pca.transform(all_my_vectors)
    index1 = 0
    plt.title(f'Random - Niv Sapir')
    for (x_point, y_point) in transformed:
        plt.scatter(x_point, y_point, s=8)
        plt.text(x_point, y_point, f'{tweets[index1].category}-{tweets[index1].index}', fontsize=7)
        index1 += 1
    plt.show()

    all_my_vectors = []
    for tweet in tweets:
        all_my_vectors.append(create_vector(3, model, tweet, all_scores))
    pca.fit(all_my_vectors)
    transformed = pca.transform(all_my_vectors)
    index1 = 0
    plt.title(f' My weight function- Niv Sapir')
    for (x_point, y_point) in transformed:
        plt.scatter(x_point, y_point, s=8)
        plt.text(x_point, y_point, f'{tweets[index1].category}-{tweets[index1].index}', fontsize=7)
        index1 += 1
    plt.show()




if __name__ == "__main__":
    kv_file = argv[1]
    xml_dir = argv[2]          # directory containing xml files from the BNC corpus (not a zip file)
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]

    # save_glove_text_file_once(kv_file)

    model: KeyedVectors = KeyedVectors.load(kv_file, mmap='r')
    output = ""
    output += task_a(model)
    output += "\n"

    corpus = Corpus()
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    n_gram = 3
    n_gram_model = NGramModel(n_gram, corpus)

    output += "\n"
    output += task_b(lyrics_file, model, n_gram_model)

    task_c(tweets_file, model)

    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output)

    print("HW4 is done finally")






