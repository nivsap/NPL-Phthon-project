import math
from sys import argv


def print_cky_tree(chart, size):
    output = ""
    start_pos = chart[0][size - 1]
    if 'S_toKey' in start_pos:
        output += 'Parsing:\n'
        output += 'S'
        spaces = 1
        pointers_backwards = []
        for item in start_pos['S_toKey'].items():
            build_tree = Print_helper(item[0], item[1], spaces)
            pointers_backwards.append(build_tree)

        while len(pointers_backwards) > 0:
            our_state = pointers_backwards.pop(0)
            if our_state.leaf == "leaf":
                output += ' >  '
                output += our_state.state
            else:
                output += '\n'
                for _ in range(our_state.spaces):
                    output += '   '
                output += our_state.state
                if our_state.leaf is not None:
                    name = str(our_state.state) + "_toKey"
                    next_states = chart[our_state.leaf[0]][our_state.leaf[1]][name]
                if not isinstance(next_states, str):
                    for s, l in next_states.items():
                        holder = Print_helper(s, l, our_state.spaces + 1)
                        pointers_backwards.insert(0, holder)
                else:
                    holder = Print_helper(next_states, "leaf", spaces+1)
                    pointers_backwards.insert(0, holder)

        output += '\nLog probability: '
        output += str(round(math.log(start_pos["S"]), 5))
        output += "\n\n"
    else:
        output += '*** This sentence is not a member of the language generated by the grammar ***\n\n'
    return output


class Convention:
    def __init__(self, probability, from_rule, to_rule: list, terminal):
        self.to_rule = to_rule
        self.from_rule = from_rule
        self.probability = probability
        self.terminal = terminal
# Implement your code here #

class Print_helper:
    def __init__(self, state, leaf, spaces):
        self.state = state
        self.leaf = leaf
        self.spaces = spaces


def calculate_probability(our_conventions, left, right, chart, i, j, stop_flag):
    conventions = []
    for conv in our_conventions:
        if set({left, right}).issubset(set(conv.to_rule)):
            conventions.append(conv)
    for conv in conventions:
        prob = conv.probability * chart[i][stop_flag][left] * chart[stop_flag + 1][j][right]
        if prob > chart[i][j].get(conv.from_rule, -1):
            chart[i][j][conv.from_rule] = prob
            name = str(conv.from_rule) + "_toKey"
            chart[i][j][name] = {left: [i, stop_flag], right: [stop_flag + 1, j]}


def calculate_the_diagonal(our_word, our_conventions, chart, j):
    conventions = []
    for conv in our_conventions:
        if set([our_word]).issubset(set(conv.to_rule)):
            conventions.append(conv)
    for conv in conventions:
        chart[j][j][conv.from_rule] = conv.probability
        conv.terminal = True
        name = str(conv.from_rule) + "_toKey"
        chart[j][j][name] = our_word


def search_nodes(nodes, chart, i, stop_flag, j):
    to_return_nodes = []
    for n in nodes:
        if i >= 0:
            if n in chart[i][stop_flag]:
                to_return_nodes.append(n)
        else:
            if n in chart[stop_flag+1][j]:
                to_return_nodes.append(n)
    return to_return_nodes


def cky_task_a(sentences, our_conventions, our_states):
    output = ''
    for sen in sentences:
        sen = sen.split()
        temp_output = ''
        for i in sen:
            temp_output += str(i)
            temp_output += ' '

        output += 'Sentence: '
        output += temp_output
        output += "\n"
        chart = [[{} for _ in range(len(sen))] for _ in range(len(sen))]
        index = 0
        for j in range(len(sen)):
            our_word = sen[j]
            calculate_the_diagonal(our_word, our_conventions, chart,j)
            for i in range(j - 1, -1, -1):
                for stop_flag in range(i, j):
                    set_nodes_1 = search_nodes(our_states, chart, i, stop_flag, -1)
                    for left in set_nodes_1:
                        set_nodes_2 = search_nodes(our_states, chart, -1, stop_flag, j)
                        for right in set_nodes_2:
                            calculate_probability(our_conventions, left, right, chart, i, j, stop_flag)
        output += print_cky_tree(chart, len(sen))
    return output


if __name__ == '__main__':

    input_grammar = argv[1]         # The name of the file that contains the probabilistic grammar
    input_sentences = argv[2]       # The name of the file that contains the input sentences (tests)
    output_trees = argv[3]          # The name of the output file

    # Implement your code here #
    #our_states = ['S', 'NP', 'VP', 'D', 'N', 'PP', 'P', 'V', 'a', 'the', 'this', 'that', 'some', 'flight', 'city', 'meal', 'airport', 'pilot', 'stop', 'cake', 'cream', 'passenger', 'stops', 'from', 'in', 'on', 'to', 'with', 'flies', 'goes', 'left', 'says', 'serves']
    our_conventions = []
    our_states = []
    with open(input_grammar) as conventions:
        for sen in conventions:
            split_sen = []
            for word in sen.split(' '):
                split_sen.append(word.strip().replace('\n', ''))
            probability_of_grammar = float(split_sen[0])
            from_rule = split_sen[1]
            to_rule = split_sen[3:]
            convention = Convention(probability_of_grammar, from_rule, to_rule, False)
            our_conventions.append(convention)
    for conv in our_conventions:
        rules = []
        rules.extend(conv.to_rule)
        rules.extend(conv.from_rule)
        for rule in rules:
            if rule not in our_states:
                our_states.append(rule)

    sentences_file = open(input_sentences)
    sentences_of_me = []
    for sen in sentences_file:
        sentences_of_me.append(sen)
    sentences_file.close()

    output_hw5 = cky_task_a(sentences_of_me, our_conventions, our_states)

    with open(output_trees, 'w', encoding='utf8') as output_file:
        output_file.write(output_hw5)

    print("NLU HW IS DONE")
