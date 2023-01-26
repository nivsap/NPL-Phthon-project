from sklearn.metrics import classification_report
import random
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import cross_val_score, train_test_split
from sys import argv
from collections import defaultdict
import gender_guesser.detector as gender
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Your implemented classes from Ex1, you may change them here according to your needs:
class Chunk:
    def __init__(self, sentences):
        self.sentences = sentences
        all_genders = [s.author_gender for s in sentences]
        if len(set(all_genders)) == 1:
            self.overall_gender = all_genders[0]
        else:
            self.overall_gender = "Unknown"


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

    def __init__(self, tokens, t, n, author_gender, authors):
        self.tokens = tokens  # arr of tokens
        self.title = t
        self.n = 1
        self.author_gender = author_gender
        self.authors = authors
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
        author_gender = ""
        tree = ET.parse(file_name)
        authors_arr = []
        for author in tree.iter(tag='author'):
            authors_arr.append(author.text)
        if len(authors_arr) == 1:
            try:
                if len(authors_arr[0].split(',')) > 1:
                    author_temp = authors_arr[0].split(',')[1].strip()
                    author_gender = gender.Detector().get_gender(author_temp)
            except IndexError:
                print("index out of bounds")
        else:
            author_gender = "Unknown"
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
            new_sentence = Sentence(tokens_arr, "", int(sentence.attrib['n']),author_gender,authors_arr)
            self.sentences.append(new_sentence)
        return

    def union_sentences_to_chunks(self):
        chunks = []
        ten_sentences = []
        sentences_counter = 0
        counter = 1
        for sentence in self.sentences:
            if counter % 10 != 0:
                ten_sentences.append(sentence)
            else:
                ten_sentences.append(sentence)
                chunks.append(Chunk(ten_sentences))
                self.gender_of_chunks.append(sentence.author_gender)
                ten_sentences = []
            counter = counter + 1
        self.chunks = chunks


# Implement a "Classify" class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class). Make sure that the class contains the relevant fields for
# classification, and the methods in order to complete the tasks:


class Classify:

    def __init__(self, corpus):
        self.corpus = corpus
        male_temp, female_temp = self.insert_gender_chunk(corpus)
        self.male_chunks = male_temp
        self.male_chunks_size = len(self.male_chunks)
        self.female_chunks = female_temp
        self.female_chunks_size = len(self.female_chunks)

    def insert_gender_chunk(self, corpus):
        male_chunks = []
        female_chunks = []
        i = 0
        while i < len(corpus.chunks):
            if corpus.gender_of_chunks[i] == "male":
                male_chunks.append(corpus.chunks[i])
            elif corpus.gender_of_chunks[i] == "female":
                female_chunks.append(corpus.chunks[i])
            i += 1
        return male_chunks, female_chunks

    def down_sampling_classes(self):
        target = min(self.female_chunks_size, self.male_chunks_size)
        if self.female_chunks_size > self.male_chunks_size:
            self.female_chunks = random.sample(self.female_chunks, target)
            self.female_chunks_size = target
        else:
            self.male_chunks = random.sample(self.male_chunks, target)
            self.male_chunks_size = target


class Chunks:
    def __init__(self, chunks):
        self.chunks = chunks

    # magic method
    def __getitem__(self, item):
        sentences = self.chunks[item].sentences
        tokens_arr = []
        for s in sentences:
            tokens_arr.append(s.tokens)
        #tokens_list = [sen.tokens for sen in sentences]
        tokens = np.concatenate(tokens_arr)
        words = []
        for word in tokens:
            words.append(word.value)
        words = np.array(words)
        return ' '.join(words)


def classifiction(classify, vector_method):
    chunks = Chunks(np.concatenate([classify.female_chunks, classify.male_chunks]))
    train_data = []
    if vector_method == "BOW":
        vector_tfi = TfidfVectorizer()
        train_data = vector_tfi.fit_transform(chunks)
    else:
        dict_of_words = defaultdict(dict)
        male_dict = {}
        female_dict = {}
        for chunk in classify.male_chunks:
            for sen in chunk.sentences:
                for token in sen.tokens:
                    if male_dict.get(token.value) is None:
                        male_dict[token.value] = 1
                    else:
                        male_dict[token.value] += 1

        for chunk in classify.female_chunks:
            for sen in chunk.sentences:
                for token in sen.tokens:
                    if female_dict.get(token.value) is None:
                        female_dict[token.value] = 1
                    else:
                        female_dict[token.value] += 1
        # give each word a score
        scores_dict = {}
        temp_dict = {}
        temp_dict = male_dict.copy()
        temp_dict.update(female_dict)
        for word in temp_dict:
            scores_dict[word] = abs(male_dict.get(word, 0) / female_dict.get(word, 1))
        dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
        # scores_dict = np.array(scores_dict)
        count_vector = CountVectorizer()
        count_vector.fit_transform(scores_dict)
        train_data = count_vector.transform(chunks)
        train_labels = np.array([c for c in scores_dict.keys()])


    temp = []
    i = 0
    while i < len(chunks.chunks):
        if i < len(chunks.chunks)/2:
            temp.append("female")
        else:
            temp.append("male")
        i += 1

    train_labels = np.array([c for c in temp])
    # model 10 folds
    model_10_folds = KNeighborsClassifier()
    model_10_folds.fit(train_data, train_labels)
    cross_val_acc_model = cross_val_score(model_10_folds, train_data, train_labels, cv=10)
    predicted = model_10_folds.predict(train_data)
    report = classification_report(train_labels, predicted, target_names=["male", "female"])

    # model 7:3 split
    model_split = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3)
    model_split.fit(X_train, y_train)
    accuracy = model_split.score(X_test, y_test)

    predicted2 = model_split.predict(train_data)
    report2 = classification_report(train_labels, predicted2, target_names=["male", "female"])

    return cross_val_acc_model, report, accuracy, report2


def create_text(file, file_name):
    output_file = open(file_name, 'w', encoding='utf8')
    output_file.write(file)
    return


def main(xml_dir, output_file):
    corpus = Corpus()
    xml_files = os.listdir(xml_dir)
    for file in xml_files:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    corpus.sentences = np.array(corpus.sentences)
    corpus.union_sentences_to_chunks()

    classify = Classify(corpus)
    male_count, female_count = classify.male_chunks_size, classify.female_chunks_size
    output_str = "Before Down-sampling:\nFemale: "
    output_str += str(female_count)
    output_str += " Male: "
    output_str += str(male_count)
    output_str += "\n"
    classify.down_sampling_classes()
    male_count, female_count = classify.male_chunks_size, classify.female_chunks_size
    output_str += "After Down-sampling:\nFemale: "
    output_str += str(female_count)
    output_str += " Male: "
    output_str += str(male_count)
    output_str += "\n\n"

    output_str += "== BoW Classification =="
    output_str += "\n\n"
    cross_val_score_list, cross_val_report, reg_val_score, reg_val_report = classifiction(classify, "BOW")
    output_str += "Model 10 folds: "
    output_str += "\n"
    output_str += "Cross Validation Accuracy: "
    output_str += str(cross_val_score_list[-1])
    output_str += "\n"
    output_str += str(cross_val_report)
    output_str += "\n"
    output_str += "Model split val 3:7 : "
    output_str += "\n"
    output_str += "3:7 split Accuracy: "
    output_str += str(reg_val_score)
    output_str += "\n"
    output_str += str(reg_val_report)
    output_str += "\n"

    output_str += "\n"
    output_str += "== Custom Feature Vector Classification =="
    output_str += "\n\n"
    output_str += "Model 10 folds: "
    output_str += "\n"
    cross_val_score_list, cross_val_report, reg_val_score, reg_val_report = classifiction(classify, "my vector")
    output_str += "Cross Validation Accuracy: "
    output_str += str(cross_val_score_list[-1])
    output_str += "\n"
    output_str += str(cross_val_report)
    output_str += "\n"
    output_str += "Model split val 3:7 : "
    output_str += "\n"
    output_str += "3:7 split Accuracy: "
    output_str += str(reg_val_score)
    output_str += "\n"
    output_str += str(reg_val_report)

    create_text(output_str, output_file)

    print("The program ended successfully")


if __name__ == '__main__':
    xml_dir = argv[1]        # directory containing xml files from the BNC corpus (not a zip file)
    output_file = argv[2]
    main(xml_dir, output_file)
