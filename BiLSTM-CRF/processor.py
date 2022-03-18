import json


def transfer_data_5(source_file, target_file):
    with open(source_file, encoding="utf-8-sig") as f, open(target_file, "w+", encoding="utf-8") as g:
        length = 0
        error = 0
        for line in f:
            try:
                line_json = json.loads(line)
                entity_list = []
                text = line_json["originalText"]
                for entities in line_json["entities"]:
                    entity_list.append({"entity_index": {"begin": entities["start_pos"],
                                                         "end":  entities["end_pos"]},
                                                "entity_type": entities["label_type"],
                                        "entity": text[entities["start_pos"]:entities["end_pos"]]})
                g.write(json.dumps({"text": text, "entity_list": entity_list}, ensure_ascii=False) + "\n")
                length += 1
            except:
                error += 1
        print("错误：{}个".format(error))
        print("共有{}行".format(length))

def transfer_data_test(source_file, target_file):
    with open(source_file, encoding="utf-8-sig") as f, open(target_file, "w+", encoding="utf-8") as g:
        length = 0
        error = 0
        for line in f:
            try:
                line_json = json.loads(line)
                entity_lists = []
                text = line_json["text"]#测试集需要将"originalText"换成text 仔细观察会发现测试集格式和训练集有所不同
                for entities in line_json["entity_list"]:#测试集需要换成entity_list
                    entity_lists.append({"entity_index": {"begin": entities["entity_index"]['begin'],
                                                         "end":  entities["entity_index"]['end']},
                                                "entity_type": entities["entity_type"],
                                        "entity": text[entities["entity_index"]['begin']:entities["entity_index"]['end']]})
                g.write(json.dumps({"text": text, "entity_list": entity_lists}, ensure_ascii=False) + "\n")
                length += 1
            except:
                error += 1
        print("错误：{}个".format(error))
        print("共有{}行".format(length))

transfer_data_5("./data/subtask1_training_part1.txt", "./data/train.json")
transfer_data_5("./data/subtask1_training_part2.txt", "./data/dev.json")
transfer_data_test("./data/test.txt", "./data/test.json")