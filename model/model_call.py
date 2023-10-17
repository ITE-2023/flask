import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from BERTClassifier import BERTClassifier
from KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer
from BERTDataset import BERTDataset

PATH = os.path.abspath(__file__)[:-20]

device = torch.device("cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
weight = os.path.join(PATH,"model/train/model_state_dict.pt")
model.load_state_dict(torch.load(weight, map_location=device), strict=False)

# 파라미터 설정
max_len = 256
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5


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
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")


if __name__ == "__main__":
    end = 1
    while end == 1 :
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0" :
            break
        predict(sentence)
        print("\n")