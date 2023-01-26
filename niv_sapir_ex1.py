import operator
import os
import re
import string
from collections import defaultdict
from sys import argv

sentence_index = 1

class Token:
    def __init__(self, t, c, p, h, nam, o, v):
        self.type = t  #word or char
        self.c5 = c    #c5 tag
        self.pos = p   #pos = part of speech
        self.hw = h    #head word
        self.name = nam   #usually empy in our corpus
        self.occurs = o   #numbers of occurs to each word
        self.value = v  #value of token
        return

    def __hash__(self):
        return hash(set(self.type, self.c5, self.pos, self.hw, self.name, self.occurs, self.value, self.level))


class Sentence:
    def __init__(self, tokens, t, n):
        self.tokens = tokens  #arr of tokens
        self.title = t
        self.n = 1
        return


def text_clean(text):
    """
    Cleans the text from some characters.
    Remove [#] notations on words if exsits.
    Reduce double or more '=' signs to one.
    Remove newlines.
    Remove tabs.
    Remove double spaces.
    Remove special space char.
    :param text: Text.
    :return: Clean text.
    """
    text = re.sub(b'=+', b'=', text)
    text = re.sub(b"\[\d+\]", b"", text)
    text = re.sub(b"\n", b" ", text)
    text = re.sub(b"\t+", b" ", text)
    text = re.sub(b" +", b" ", text)
    text = re.sub("\u200b".encode(), b"", text)
    return text.decode()


def split_to_sentences(text):
    """
    Split text to arrays of sentences.
    Sentences difne by '.' '?' '!'
    :param text: Text.
    :return: list of sentences.
    """
    sentences_list = []
    sentence_list = []
    parentheses = defaultdict(int)
    parentheses["count"] = 0
    flag = False
    for i, c in enumerate(text):
        if c == '(':
            parentheses[c] += 1
        elif c == ')':
            parentheses['('] -= 1
        elif c == '"':
            if parentheses[c] > 0:
                parentheses[c] -= 1
            else:
                parentheses[c] += 1
        if is_end_of_sentence(i, c, text, parentheses):
            sentence_list.append(c)
            sentences_list.append("".join(sentence_list).strip())
            sentence_list.clear()
            parentheses.clear()
            flag = True
        elif flag:
            flag = False
            if c == " ":
                continue
            sentence_list.append(c)
        else:
            sentence_list.append(c)
    if sentence_list:
        sentences_list.append("".join(sentence_list).strip())
    return sentences_list


def arranging_titles(sentences):
    """
    arranging title to "= title =" as a sentence
    :param sentences: list of sentences.
    :return: list of sentences.
    """
    pattern = r'\=(.*?)\='
    new_sentences = []
    for sentence in sentences:
        while sentence.startswith('='):
            str = re.search(pattern, sentence)
            sentence = re.sub(pattern, '', sentence, 1).lstrip()
            try:
                new_sentences.append(str.group(0))
            except:
                break
        new_sentences.append(sentence.lstrip())
    return new_sentences


def is_numbers(before, after):
    # Check if chars are numbers
    return str.isdigit(before) and str.isdigit(after)


def is_letters(before, after):
    # Check if chars are letters.
    return str.isalpha(before) and str.isalpha(after)


def is_numbers_or_letters(before, after):
    # Check if we have mix between number and letter.
    if before.isalnum() and after.isalnum():
        return True
    else:
        return False


def is_shortcut(index, char, text):
    """
    Check if given char is shortcut word.
    :param char: char.
    :param text: text.
    :param index: index of given char in text.
    :return: True or False.
    """
    if (len(text)) <= (index + 2):
        return False
    after_char = text[index + 1]
    before_char = text[index - 1]
    two_index_after_char = text[index + 2]
    if (char == "."):
        # regular sentence
        if before_char.isalpha() and (after_char == " ") and two_index_after_char.isalpha():
            if two_index_after_char.islower():
                return True
        # case "something somewhere 1.6.2 and even more..
        if before_char.isdigit() and (after_char == " ") and two_index_after_char.isalpha():
            if two_index_after_char.islower():
                return True
        # e.g. shortcut.
        if before_char == 'g' and text[index - 2] == '.' and text[index - 3] == 'e':
            return True
        # shortcut to vs.
        if before_char == 'v' and text[index - 2] == ' ':
            return True
        # shortcut to for Mr.
        if before_char == 'r' and text[index - 2] == 'M':
            return True
        # shortcut to Dr.
        if before_char == 'r' and text[index - 2] == 'D':
            return True
        # i.e. shortcut.
        if before_char == 'e' and text[index - 2] == '.' and text[index - 3] == 'i':
            return True
        # shortcut to _p.
        if before_char == 'p' and text[index - 2] == ' ':
            return True
        # name shortcut
        if (char == ".") and before_char.isupper() and (after_char == " ") and two_index_after_char.isalpha():
            if two_index_after_char.isupper():
                return True
    return False


def clean_whitespace(sentences):
    # lstrip() = Remove spaces to the left of the string.
    temp_sentences = []
    for sentence in sentences:
        if not (
                sentence == '.' or sentence == '"' or sentence == ')' or sentence == '".' or sentence == "..." or sentence == ').'):
            if sentence.startswith(",") or sentence.startswith(')') or sentence.startswith('),'):
                sentence = sentence[1:].lstrip()
            temp_sentences.append(sentence.lstrip())
    return temp_sentences


def is_end_of_sentence(index, char, text, helper):
    """
    Check if given char is an end of a sentence.
    End of sentence == ?/./!/..+
    :param index: the index of the char in the text string.
    :param char: char in sentence.
    :param text: text.
    :return: True or False
    """
    if (len(text)) <= (index + 2):
        return False
    before_char = text[index - 1]
    after_char = text[index + 1]
    if char == '.':
        if is_numbers(before_char, after_char) or is_letters(before_char, after_char) or is_numbers_or_letters(
                before_char, after_char):
            return False
        elif is_shortcut(index, char, text):
            return False
        elif after_char == '.':  # handel case "etc..."
            return False
        elif after_char == '-':
            return False
        else:
            return True
    if char == '"' or char == ')':
        if before_char == '.' or before_char == '?' or before_char == '!':  # or before_char == ';':
            return True
    return (char == '?') or (char == '.') or (char == '!')  # or(char == ';')


def tokenize_words(sentences):
    """
    tokenize the words in the sentences.
    :param sentences:
    :return: list of sentences that includes tokens.
    """
    many_tokens = []
    for sentence in sentences:
        sentence = tokenize_special_characters(sentence)
        tokens = []
        if sentence.startswith(" "):
            sentence = sentence[1:]
        words = sentence.split(" ")
        # create tokens
        for word in words:
            if (word in string.punctuation):
                word_or_char = False
            else:
                word_or_char = True
            new_token = Token('word' if word_or_char else 'char', "", "", "", "", "", word + " ")
            tokens.append(new_token)
        many_tokens.append(tokens)
    return many_tokens


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


class Corpus:
    def __init__(self):
        self.sentences = []   #arr of sentences

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the XML file that will be read
        :return: None
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_name)
        root = tree.getroot()
        dict_value = dict()
        # dictionary of word to count ranks and frequencies
        for w in root.iter('w'):
            if (dict_value.get(w.text) == None):
                dict_value[w.text] = 1
            else:
                dict_value[w.text] += 1
        # sort dict by frequencies to aprove zipf's law
        dict_value = sorted(dict_value.items(), key=operator.itemgetter(1))
        #print(dict_value)
        # iterate over sentence of the xml files
        for sentence in tree.iter('s'):
            tokens_arr = []
            for w in sentence:  # iterate over words in sentence.
                if w.tag in ('w', 'c'):  # inesrt word or char as a token
                    token = Token(w.tag, w.attrib.get('c5', ""), w.attrib.get('pos', ""), w.attrib.get('hw', ""), "",
                                  "", w.text)
                    tokens_arr.append(token)
            new_sentence = Sentence(tokens_arr, "", int(sentence.attrib['n']))
            self.sentences.append(new_sentence)
        return

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an text file (from Wikipedia), read the content
        from it and add it to the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the text file that will be read
        :return: None
        """
        global sentence_index
        with open(file_name, "rb") as text:
            wiki_text = text_clean(text.read())
            sentences = split_to_sentences(wiki_text)
            sentences = arranging_titles(sentences)
            sentences = clean_whitespace(sentences)
            tokens_arr = tokenize_words(sentences)
            for t in tokens_arr:
                new_sentence = Sentence(t, "", sentence_index)
                sentence_index += 1
                self.sentences.append(new_sentence)
        return

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param file_name: The name of the file that the text will be written on
        :return: None
        """
        flag = 0
        file_str = ''
        for sentence in self.sentences:
            for token in sentence.tokens:
                if ("=" in token.value):
                    flag += 1
                file_str += token.value
                if (flag == 2):
                    sentence.title = "true"
                    file_str += '\n'
                    flag = 0
            if (flag != 2):  # get '\n' each sentence and each title
                file_str += '\n'
            flag = 0
        output_file = open(file_name, 'w', encoding='utf8')
        output_file.write(file_str)
        return


if __name__ == '__main__':

    xml_dir = argv[1]  # directory containing xml files from the BNC corpus (not a zip file)
    wiki_dir = argv[2]  # directory containing text files from Wikipedia (not a zip file)
    output_file = argv[3]

    corpus = Corpus()
    xml_files = os.listdir(xml_dir)
    for file in xml_files:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))

    wiki_files_names = os.listdir(wiki_dir)
    for file in wiki_files_names:
        corpus.add_text_file_to_corpus(os.path.join(wiki_dir, file))

    corpus.create_text_file(output_file)
    print("end of the road")