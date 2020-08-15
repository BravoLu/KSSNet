from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                   'hair drier', 'toothbrush']

word_embeddings = []

for c in classes:
    inputs = tokenizer(c, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs[0]
    #import pdb
    #pdb.set_trace()
    word_embedding = last_hidden_states[0].detach().numpy()
    #print(word_embedding.size())
    word_embedding = np.mean(word_embedding, axis=0)
    print('{} {}'.format(c, word_embedding.shape))
    word_embeddings.append(word_embedding)

word_embeddings = np.stack(word_embeddings, axis=0)
print('{}'.format(word_embeddings.shape))
import pickle

with open('bert_coco.pkl', 'wb') as f:
    pickle.dump(word_embeddings, f)

print("{}",format(type(word_embeddings)))
#print(word_embeddings.shape)
with open('data/coco_glove_word2vec.pkl', 'rb') as f:
    d = pickle.load(f)

print("{} {}".format(d.shape, type(d)))
