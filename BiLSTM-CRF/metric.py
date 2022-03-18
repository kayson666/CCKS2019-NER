import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        # seq = [item for sublist in seq for item in sublist + ['S']]
        seq = [item for sublist in seq for item in sublist]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):#判断是前一个字符是否为结尾
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):#判断当前字符是否为开始
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False#默认不是结尾
    #前面那个标签是S即为实体结尾
    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形 只要当前标签是BSO，前面那个是实体结尾
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    #b-company b-org不规则出现或b-company i-org
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False#默认不是开始
    #当前标签是BS即开始
    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True
    #不规则
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    #不规则的 当前不是O，必是其它，而且前后标签不一致
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):

    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}#计算每个类别的F1值
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w',encoding='utf-8')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")