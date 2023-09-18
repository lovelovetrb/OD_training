# csvファイルを読み込み、指定した列の値を抽出し、csvファイルに書き込む
import functools
import logging
import random

import pandas as pd
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import \
    concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import (split_newline,
                                                         split_punctuation)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def splitSentence(text):
    split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
    concat_tail_no = functools.partial(
        concatenate_matching,
        former_matching_rule=r"^(?P<result>.+)(の)$",
        remove_former_matched=False,
    )
    segmenter = make_pipeline(normalize, split_newline, concat_tail_no, split_punc2)

    # Golden Rule: Simple period to end sentence #001 (from https://github.com/diasks2/pragmatic_segmenter/blob/master/spec/pragmatic_segmenter/languages/japanese_spec.rb#L6)
    result = list(segmenter(text))
    print(result)
    return result


# csvファイルを読み込む
logging.info("load csv file")
FILE_NAME = "test"
df = pd.read_csv(f"data/{FILE_NAME}.csv", encoding="utf-8")

data_list = df.values.tolist()
logging.info(f"csv file length: {len(data_list)}")
parsed_data_list = []
new_sentence_num = 0
ai_answer_num = 0
human_answer_num = 0

for index, data in enumerate(data_list):
    if index % 100 == 0:
        logging.info(f"{index} / {len(data_list)}")
    label = data[0]
    question = data[1]
    answer = data[2]

    answer_list = splitSentence(answer)
    new_sentence_num += len(answer_list)

    if label == 0:
        ai_answer_num += len(answer_list)
    else:
        human_answer_num += len(answer_list)

    for splited_answer in answer_list:
        parsed_data_list.append([label, question, splited_answer])
logging.info(f"finished {len(data_list)} -> {new_sentence_num}")
logging.info(f"ai_answer_num: {ai_answer_num}")
logging.info(f"human_answer_num: {human_answer_num}")


# パースした結果をシャッフルする
logging.info("shuffle")
random.shuffle(parsed_data_list)

# パースした結果をcsvファイルに書き込む
logging.info("write csv file")
df = pd.DataFrame(parsed_data_list, columns=["label", "question", "answer"])
df.to_csv(f"data/{FILE_NAME}_parsed.csv", encoding="utf-8", index=False)

logging.info("finish")
