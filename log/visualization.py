"""
visualize the log in log.txt
"""

import definitions
import os
from datetime import datetime
from prettytable import PrettyTable


def get_data():
    blocks = []
    if not os.path.exists(definitions.ROOT_DIR + 'log/log.txt'):
        raise Exception('No data at all.You should run "python main.py"')
    with open(definitions.ROOT_DIR + 'log/log.txt', 'r') as f:
        lines = [line.strip('\n') for line in f.readlines()]

        for i, line in enumerate(lines):
            if 'program start' in line:
                break
        lines = lines[i:]

        flag = False
        temp_block = []
        for line in lines:
            if flag:
                temp_block.append(line)
            if 'program start' in line and temp_block:
                temp_block.clear()
                flag = True
            elif 'program start' in line:
                flag = True
            elif 'program end' in line:
                blocks.append(temp_block.copy())
                temp_block.clear()
                flag = False

    assert blocks
    data = []
    for block in blocks:
        date = datetime.strptime(block[0].split(' - tensorflow - ')[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
        layers = eval(block[0].split(': ')[1])
        filters = eval(block[1].split(': ')[1])
        fc = eval(block[2].split(': ')[1])
        lr = float(block[3].split(': ')[1])
        keep_prob = eval(block[4].split(': ')[1])
        expand_data = eval(block[5].split(': ')[1])
        mini_batch_size = eval(block[6].split(': ')[1])
        info = {'date': date,
                'layers': layers,
                'filters': filters,
                'fc': fc,
                'lr': lr,
                'keep_prob': keep_prob,
                'expand_data': expand_data,
                'mini_batch_size': mini_batch_size}

        process = []
        for i in range(len(block)):
            if 'Iteration' in block[i]:
                x = int(block[i].split(' ')[1])
                cost = float(block[i + 1].split(': ')[1])
                train_acc = float(block[i + 2].split(': ')[1])
                test_acc = float(block[i + 3].split(': ')[1])
                process.append((x, (cost, train_acc, test_acc)))
                i += 4
        data.append((info, process))
    return data


def get_structure(layers, filters, fc):
    result = ''
    for i, layer in enumerate(layers):
        if layer != 0 and layer != -1:
            result += str(layer) + '*' + str(filters[i][:2]) + ' - '
    for layer in fc:
        result += layer + ' - '

    return result + '1'


if __name__ == '__main__':
    data = get_data()
    table = PrettyTable(
        ['index', 'structure', 'learning rate', 'keep prob', 'expand', 'train accuracy', 'test accuracy'])
    for i, item in enumerate(data):
        info = item[0]
        process = item[1]
        structure = get_structure(info['layers'], info['filters'], info['fc'])
        table.add_row(
            [str(i + 1), structure, '%f' % info['lr'], info['keep_prob'], info['expand_data'], process[-1][1][1],
             process[-1][1][2]])
    print(table)
