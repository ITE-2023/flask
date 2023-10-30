import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from BERTClassifier import BERTClassifier
from KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer
from BERTDataset import BERTDataset

import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

PATH = os.path.abspath(__file__)[:-20]

device = torch.device("cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
weight = os.path.join(PATH,"model/train/model_state_dict_231018.pt")
model.load_state_dict(torch.load(weight, map_location=device), strict=False)

# 파라미터 설정
max_len = 256
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5


def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


def predict(predict_sentence):
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

            emotion = np.argmax(logits)
            probability.append(emotion)

    return probability

openai_key = os.environ.get("MY_API_KEY")

#Getting Embeddings
def get_embedding(content, openai_key):
    openai.api_key = openai_key
    # JSON 데이터 생성
    data = {
        "text": content
    }
    # JSON 데이터를 문자열로 변환
    json_data = json.dumps(data)

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=json_data
    )

    embedding = response['data'][0]['embedding']

    return embedding

#cosine_similarity 함수
def cosineSimilarity(report_text, music_file_lyrics):
    content1 = report_text
    #dataframe -> list 변환
    music_file_lyrics_list = music_file_lyrics.values.tolist()
    cosine_similarity_list = []

    text1_embs = get_embedding(content1, openai_key)

    for rank, lyric in music_file_lyrics_list:
        if lyric != None:
            text2_embs = get_embedding(lyric, openai_key)

            cosine_sim = cosine_similarity([text1_embs], [text2_embs])[0][0]
            cosine_similarity_list.append([rank, cosine_sim])
    #유사도를 기준으로 내림차순
    cosine_similarity_list.sort(key=lambda x: -x[1])


    return cosine_similarity_list



if __name__ == "__main__":
    end = 1
    while end == 1 :
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0":
            break
        report_text = sentence
        emotion = predict(sentence)[-1]
        if emotion == 0:
            print("슬픔")
            music_file_lyrics = pd.read_excel("dataset/sad.xlsx", usecols=["순위", "가사"]).replace({np.nan: None}) #NaN을 None으로 변환
            text_sim = cosineSimilarity(report_text, music_file_lyrics)
        elif emotion == 1:
            print("공포")
            music_file_lyrics = pd.read_excel("dataset/scary.xlsx", usecols=["순위", "가사"]).replace({np.nan: None})
            text_sim = cosineSimilarity(report_text, music_file_lyrics)
        elif emotion == 2:
            print("분노")
            music_file_lyrics = pd.read_excel("dataset/anger.xlsx", usecols=["순위", "가사"]).replace({np.nan: None})
            text_sim = cosineSimilarity(report_text, music_file_lyrics)
        elif emotion == 3:
            print("놀람")
            music_file_lyrics = pd.read_excel("dataset/surprised.xlsx", usecols=["순위", "가사"]).replace({np.nan: None})
            text_sim = cosineSimilarity(report_text, music_file_lyrics)
        elif emotion == 4:
            print("행복")
            music_file_lyrics = pd.read_excel("dataset/happy.xlsx", usecols=["순위", "가사"]).replace({np.nan: None})
            text_sim = cosineSimilarity(report_text, music_file_lyrics)
        print("\n")

