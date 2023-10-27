import os

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from .BERTClassifier import BERTClassifier
from .KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer
from .BERTDataset import BERTDataset

PATH = os.path.abspath(__file__)[:-23]

device = torch.device("cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
weight = os.path.join(PATH, "model/train/model_state_dict_231018.pt")
model.load_state_dict(torch.load(weight, map_location=device), strict=False)

# 파라미터 설정
max_len = 256
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

happy = pd.read_excel(PATH + "dataset/happy.xlsx")
anger = pd.read_excel(PATH + "dataset/anger.xlsx")
sad = pd.read_excel(PATH + "dataset/sad.xlsx")
scary = pd.read_excel(PATH + "dataset/scary.xlsx")
surprised = pd.read_excel(PATH + "dataset/surprised.xlsx")


def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


def predict(predict_sentence):
    global probability
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for logits in out:
            logits = logits.detach().cpu().numpy()
            logits = np.round(new_softmax(logits), 3).tolist()
            probability = []
            for logit in logits:
                probability.append(np.round(logit, 3))

            argmax = np.argmax(logits)
            probability.append(argmax)

    return probability


def randomMusic(emotion):
    global musicList, emotionResult
    if emotion == 0:
        musicList = sad.sample(n=3)
        emotionResult = "슬픔"
    elif emotion == 1:
        musicList = scary.sample(n=3)
        emotionResult = "공포"
    elif emotion == 2:
        musicList = anger.sample(n=3)
        emotionResult = "분노"
    elif emotion == 3:
        musicList = surprised.sample(n=3)
        emotionResult = "놀람"
    elif emotion == 4:
        musicList = happy.sample(n=3)
        emotionResult = "행복"

    musicListResult = []
    for index, row in musicList.iterrows():
        musicListResult.append(
            {
                "title": row['제목'],
                "singer": row['가수'],
                "imageUrl": row['앨범이미지']
            }
        )
    return emotionResult, musicListResult


def recommend(content):
    predict_content_ = predict(content)[-1]
    return randomMusic(predict_content_)


if __name__ == "__main__":
    end = 1
    while end == 1:
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0":
            break

        emotion = predict(sentence)[-1]
        if emotion == 0:
            print("슬픔")
        elif emotion == 1:
            print("공포")
        elif emotion == 2:
            print("분노")
        elif emotion == 3:
            print("놀람")
        elif emotion == 4:
            print("행복")
        print("\n")
