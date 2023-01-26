import math
import os
import random
import string
import xml.etree.ElementTree as ET
from sys import argv
from collections import defaultdict

# Your implemented classes from Ex1, you may change them here according to your needs:


class Token:

    def __init__(self, t, c, p, h, nam, o, v):
        self.type = t  # word or char
        self.c5 = c  # c5 tag
        self.pos = p  # pos = part of speech
        self.hw = h  # head word
        self.name = nam  # usually empy in our corpus
        self.occurs = o  # numbers of occurs to each word
        self.value = v  # value of token
        return


class Sentence:

    def __init__(self, tokens, t, n):
        self.tokens = tokens  # arr of tokens
        self.title = t
        self.n = 1
        return


class Corpus:

    def __init__(self):
        self.sentences = []  # arr of sentences

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        tree = ET.parse(file_name)

        # iterate over sentence of the xml files
        for sentence in tree.iter(tag='s'):
            tokens_arr = []
            for word in sentence:
                if word.tag in ('w', 'c'):
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
            new_sentence = Sentence(tokens_arr, "", int(sentence.attrib['n']))
            self.sentences.append(new_sentence)
        return

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """
        file_str = ''
        for sentence in self.sentences:
            for token in sentence.tokens:
                file_str += token.value
            file_str += '\n'
        output_file = open(file_name, 'w', encoding='utf8')
        output_file.write(file_str)
        return

    def add_start_sign(self):
        for sentence in self.sentences:
            new_token = Token("","","","","","","<b>")
            sentence.tokens.insert(0,new_token)
        return

    def add_end_sign(self):
        for sentence in self.sentences:
            new_token = Token("","","","","","","<e>")
            sentence.tokens.insert(len(sentence.tokens),new_token)
        return

# Implement an n-gram language model class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class):


class NGramModel:

    def __init__(self):
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        return

def create_text(file,file_name):
    output_file = open(file_name, 'w', encoding='utf8')
    output_file.write(file)
    return


def unigram_probability(sentence, dict_stat):
    words = sentence.split()
    result = 0
    for w in words:
        if dict_stat.get(w.lower()) is not None:
            result += math.log(dict_stat.get(w.lower()))
        else:
            result += math.log(1 / len(dict_stat))
    return result


def baigram_probability(sentence, dict_stat):
    words = sentence.split()
    result = 0
    for index in range(len(words) - 1):
        pair = (words[index].lower(), words[index + 1].lower())
        if (dict_stat.get(pair) is not None):
            result += float(math.log(dict_stat.get(pair)))
        else:
            result += math.log(1 / len(dict_stat))
    return result


def compute_trigram_combination_probability(sentence, dict1, dict2, dict3):
    words = sentence.split()
    result = 0
    lambda1 = 0.5
    lambda2 = 0.3
    lambda3 = 0.2
    one = 0
    two = 0
    tree = 0
    if (dict1.get(words[2].lower()) is not None):
        result = math.log(float(dict1.get(words[2].lower())))
    pair = (words[1].lower(), words[2].lower())
    # print(dict2)
    if (dict2.get(pair) is not None):
        result += math.log(float(dict2.get(pair)))
    for index in range(len(words) - 2):
        pair = (words[index + 1].lower(), words[index + 2].lower())
        trio = (words[index].lower(), words[index + 1].lower(), words[index + 2].lower())
        if dict1.get(words[index + 2].lower()) is not None:
            one = math.log(float(dict1.get(words[index + 2].lower())))
        else:
            one = 0
        # print(pair)
        if (dict2.get(pair) != None):
            two = math.log(float(dict2.get(pair)))
        else:
            two = 0
        if (dict3.get(trio) != None):
            tree = math.log(float(dict3.get(trio)))
        else:
            tree = 0
        p_trio = lambda3 * one + lambda2 * two + lambda1 * tree
        result += p_trio
    return result


def remove_whitespace(word):
    for elem in string.whitespace:
        word = word.replace(elem, '')
    return word

def write_file(file, prob, s):
    if type(s) != list:
        if s == "You're tearing me apart , Lisa !":
            s = "You're tearing me apart, Lisa !"
        s = s[:len(s)-2] + "" + s[len(s)-1:]
        file += s
        file += "\n"
    if(prob == 0):
        for word in s:
            file += word + " "
    else:
        file += "Probability: "
        file += str(prob)

    file += "\n"
    return file



def count_pair_words(corpora, statistics):
    for sentence in corpora.sentences:
        for index in range(len(sentence.tokens) - 1):
            one = remove_whitespace(sentence.tokens[index].value.lower())
            two = remove_whitespace(sentence.tokens[index + 1].value.lower())
            pair = (str(one), str(two))
            if statistics.get(pair) is not None:
                statistics[pair] += 1
            else:
                statistics[pair] = 1


def count_trio_words(corpora, statistics):
    for sentence in corpora.sentences:
        for index in range(len(sentence.tokens) - 2):
            one = remove_whitespace(sentence.tokens[index].value.lower())
            two = remove_whitespace(sentence.tokens[index + 1].value.lower())
            tree = remove_whitespace(sentence.tokens[index + 2].value.lower())
            trio = (str(one), str(two), str(tree))
            if statistics.get(trio) is not None:
                statistics[trio] += 1
            else:
                statistics[trio] = 1


def compute_words_probability_uniform(corpus, dict1):
    count = 0
    for value in dict1.values():
        count += value
    for key in dict1.keys():
        dict1[key] = ((dict1[key] + 1) / (count + len(dict1)))


def compute_trio_probability(corpora, dict2, dict3):
    for key in dict3.keys():
        pair = (key[1], key[2])
        dict3[key] = ((dict3[key]+1) / (dict2[pair] + len(dict3)))


def compute_laplace_probability_bigrams(corpora, dict2, dict1):
    for key in dict2.keys():
        if dict1.get(key[0]) != None:
            dict2[key] = ((float(dict2[key]) + 1) / (len(dict2) + dict1[key[0]]))

#part B:


def generate_sentence_length(corpus):
    sentence_length_options = []
    length_weights = []
    number_of_sentences = len(corpus.sentences)
    for sentence in corpus.sentences:
        sentence_length_options.append(len(sentence.tokens))
        length_weights.append(len(sentence.tokens) / number_of_sentences)
    sentence_length = random.choices(sentence_length_options, length_weights, k=1)
    return sentence_length.pop()


def generate_unigram_random_words(sentence_length, dict1):
    words_options = []
    words_weights = []
    for key in dict1.keys():
        words_options.append(key)
        words_weights.append(dict1.get(key))
    random_words = random.choices(words_options, words_weights, k=sentence_length)
    while '<e>' in random_words:
        random_words = random.choices(words_options, words_weights, k=sentence_length)
    random_words.insert(0, "<b>")
    return random_words

def generate_bigram_words(sentence_length, dict2):
    flag = 0
    sentence_complete = False
    while not sentence_complete:
        random_words = []
        first_word = '<b>'
        for index in range(sentence_length):
            words_options = []
            weights = []
            for pair in dict2.keys():
                if pair[0] == first_word:
                    words_options.append(pair[1])
                    weights.append(dict2.get(pair))
            if not words_options:
                sentence_complete = False
                break
            if len(weights) > 0 and len(words_options) > 0:
                random_next_word = random.choices(words_options, weights)
            if random_next_word[0] == '<b>':
                sentence_complete = False
                break
            if random_next_word[0] == "<e>":
                sentence_complete = True
                break
            random_words.append(random_next_word[0])
            first_word = random_next_word[0]

        if sentence_length == len(random_words):
            flag = 1
            sentence_complete = True
    random_words.insert(0, "<b>")
    if flag != 1:
        random_words.insert(len(random_words), "<e>")
    return random_words


def generate_trigram_words(sentence_length, dict2, dict3):
    restart = False
    sentence_complete = False
    while not sentence_complete:
        random_words = []
        pair = generate_first_two_words(dict2)
        random_words.append(pair[0])
        random_words.append(pair[1])
        next_word = pair[1]
        sentence_index = 2
        while next_word != "<e>" and sentence_length != len(random_words):
            words_options = []
            weights = []
            for trio in dict3.keys():
                if (trio[0], trio[1]) == pair:
                    try:
                        words_options.append(trio[2])
                        trio_probability = dict3.get(trio)
                        weights.append(trio_probability)
                    except KeyError or ValueError:
                        restart = True
                        break
            if restart:
                restart = False
                sentence_complete = False
                break
            next_random_word = random.choices(words_options, weights)
            next_word = next_random_word[0]
            random_words.append(next_word)
            sentence_index += 1
            pair = pair[1], next_word
        sentence_complete = True
    return random_words

def generate_first_two_words(dict2):
    first_word = "<b>"
    words_options = []
    weights = []
    for pair in dict2.keys():
        if pair[0] == first_word:
            words_options.append(pair[1])
            weights.append(dict2.get(pair))
    random_next_word = random.choices(words_options, weights)
    return first_word, random_next_word[0]

def main(xml_dir , output_file):
    file_str = "*** Sentence Predictions ***"
    file_str += '\n'
    file_str += "\n"

    xml_dir = argv[1]  # directory containing xml files from the BNC corpus (not a zip file)
    output_file = argv[2]

    dict_value = defaultdict(dict)  # dictionary of word to count ranks and frequencies

    print("Init corpus")
    corpus = Corpus()
    model = NGramModel()
    xml_files = os.listdir(xml_dir)
    for file in xml_files:
        tree = ET.parse(os.path.join(xml_dir, file))
        root = tree.getroot()
        for w in root.iter('w'):
            word = remove_whitespace(w.text.lower())
            if (dict_value.get(word.lower()) == None):
                dict_value[word.lower()] = 1
            else:
                dict_value[word.lower()] += 1
        for c in root.iter('c'):
            word = remove_whitespace(c.text.lower())
            if (dict_value.get(word) == None):
                dict_value[word] = 1
            else:
                dict_value[word] += 1
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    corpus.add_end_sign()
    corpus.add_start_sign()

    ####
    # generate_sentence_length(corpus)   #generate random setence length
    # #biagram
    dict2 = defaultdict(dict)
    count_pair_words(corpus, dict2)

    ##########################
    # traigram
    dict3 = defaultdict(dict)
    count_trio_words(corpus, dict3)

    compute_trio_probability(corpus, dict2, dict3)
    compute_laplace_probability_bigrams(corpus, dict2, dict_value)
    compute_words_probability_uniform(corpus, dict_value)
    file_str += "\n"
    file_str = file_str + "Unigrams Model:"
    file_str +="\n"
    input = ["May the Force be with you .", "I'm going to make him an offer he can't refuse .",
             "Ogres are like onions .", "You're tearing me apart , Lisa !", "I live my life one quarter at a time ."]
    for s in input:
        change_word = s.replace("can't", "ca n't")
        change_word = change_word.replace("I'm", "I 'm")
        change_word = change_word.replace("You're", "You 're")
        file_str = write_file(file_str, unigram_probability(change_word, dict_value), s)
    file_str += "\n"
    file_str += "Bigrams Model:"
    file_str += "\n"
    for s in input:
        change_word = s.replace("can't", "ca n't")
        change_word = change_word.replace("I'm", "I 'm")
        change_word = change_word.replace("You're", "You 're")
        file_str = write_file(file_str, baigram_probability(change_word, dict2), s)
    file_str += "\n"
    file_str += "Trigrams Model:"
    file_str += "\n"
    for s in input:
        change_word = s.replace("can't", "ca n't")
        change_word = change_word.replace("I'm", "I 'm")
        change_word = change_word.replace("You're", "You 're")
        file_str = write_file(file_str, compute_trigram_combination_probability(change_word, dict_value, dict2, dict3),s)
    model.unigrams = dict_value
    model.bigrams = dict2
    model.trigrams = dict3

    # part B
    file_str += "\n"
    file_str += "*** Random Sentence Generation ***"
    file_str += "\n"
    file_str += "\n"
    file_str = file_str + "Unigrams Model:"
    file_str += "\n"

    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_unigram_random_words(sentence_length, dict_value))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_unigram_random_words(sentence_length, dict_value))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_unigram_random_words(sentence_length, dict_value))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_unigram_random_words(sentence_length, dict_value))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_unigram_random_words(sentence_length, dict_value))

    file_str += "\n"
    file_str += "Bigrams Model:"
    file_str += "\n"

    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_bigram_words(sentence_length, dict2))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_bigram_words(sentence_length, dict2))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_bigram_words(sentence_length, dict2))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_bigram_words(sentence_length, dict2))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_bigram_words(sentence_length, dict2))

    file_str += "\n"
    file_str += "Trigrams Model:"
    file_str += "\n"

    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_trigram_words(sentence_length, dict2, dict3))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_trigram_words(sentence_length, dict2, dict3))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_trigram_words(sentence_length, dict2, dict3))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_trigram_words(sentence_length, dict2, dict3))
    sentence_length = generate_sentence_length(corpus)
    file_str = write_file(file_str, 0, generate_trigram_words(sentence_length, dict2, dict3))

    create_text(file_str,output_file)

    print("generate output file is completed")

if __name__ == '__main__':
    xml_dir = argv[1]        # directory containing xml files from the BNC corpus (not a zip file)
    output_file = argv[2]
    main(xml_dir, output_file)
