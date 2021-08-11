import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import numpy
import tflearn
import tensorflow
import random
import json
import os
import itertools
from math import log
import re
from tensorflow.python.framework import ops
import re
import json
import itertools
from math import log
import datetime
import webbrowser as wb


# Cập nhật thêm câu hỏi và trả lời vào tệp để training
def update_json(filepath, var1, var2):
    now = datetime.datetime.now() # Gọi timenow đặt tên cho tag
    t = now.strftime("%y-%m-%d %H:%M:%S")
    with open(filepath,'r', encoding='utf-8') as fp:
        information = json.load(fp)
    information["intents"].append({
        "tag": t,
        "patterns": [var1],
        "responses": [var2],
        "context_set": ""
    })

    with open(filepath,'w',encoding='utf-8') as fp: # Thêm dữ liệu vào tệp JSON
        json.dump(information, fp, indent=2,)
        
# Xử lý văn vản đầu vào 
    # Làm sạch , bỏ dấu văn bản
def xoa_dau_tieng_viet(s):
    s = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', s)
    s = re.sub('[éèẻẽẹêếềểễệ]', 'e', s)
    s = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', s)
    s = re.sub('[íìỉĩị]', 'i', s)
    s = re.sub('[úùủũụưứừửữự]', 'u', s)
    s = re.sub('[ýỳỷỹỵ]', 'y', s)
    s = re.sub('đ', 'd', s)
    return s

    # Tạo bộ từ điển tiếng viêt không dấu nhờ chương trình con xoa_dau_tieng_viet với tệp nguphapvietnam.txt
map_accents = {}
for word in open('nguphapvietnam.txt', encoding="utf8").read().splitlines():
  word = word.lower()                           # Bỏ viết hoa
  no_accent_word = xoa_dau_tieng_viet(word)     # Xóa dấu
  if no_accent_word not in map_accents:
    map_accents[no_accent_word] = set()
  map_accents[no_accent_word].add(word)
  
# Đọc lm
lm = {}
for line in open('thuvientiengviettonghop.txt', encoding="utf8"):
  data = json.loads(line)
  key = data['s']
  lm[key] = data
vocab_size = len(lm)
total_word = 0
for word in lm:
  total_word += lm[word]['sum']
  
# tính xác suất dùng smoothing
def get_proba(current_word, next_word):
  if current_word not in lm:
    return 1 / total_word;
  if next_word not in lm[current_word]['next']:
    return 1 / (lm[current_word]['sum'] + vocab_size)
  return (lm[current_word]['next'][next_word] + 1) / (lm[current_word]['sum'] + vocab_size)

# hàm beam search
def beam_search(words1, k=3):
  sequences = []
  for idx, word in enumerate(words1):
    if idx == 0:
      sequences = [([x], 0.0) for x in map_accents.get(word, [word])]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in map_accents.get(word, [word]):
          current_word = seq[0][-1]
          proba = get_proba(current_word, next_word)
          # print(current_word, next_word, proba, log(proba))
          proba = log(proba)
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + proba))
      # print(all_sequences) 
      all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
      sequences = all_sequences[:k]
  return sequences

# Xóa bỏ ký tự thừa , ký tự đặc biệt
def xoa_ky_tu_dac_biet(sentence):
  sentence = sentence.lower()
  sentence = re.sub(r'[.,~`!@#$%\^&*\(\)\[\]\\|:;\'"]+', ' ', sentence)
  sentence = re.sub(r'\s+', ' ', sentence).strip()
  return sentence

with open("training.json",encoding="utf8") as file:
    data = json.load(file)
try:
    with open("data.pickle", "ab") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
        
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "ab") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load(model.tflearn)
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


 
    
    
def chat():
    print("Noi gi di may?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        sentence = xoa_ky_tu_dac_biet(inp)
        _sentence = xoa_dau_tieng_viet(sentence)
        words1 = _sentence.split()
        results1 = beam_search(words1, k=5)
        inp = ' '.join(results1[0][0])
        
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(results)
        
        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    hoang = random.choice(responses)
                    if hoang == 'Google':
                        inp3 = input("Bot Dz: Bạn muốn tìm kiếm gì ? \nYou: ")
                        url=f"https://www.google.com/search?q="+inp3
                        wb.get().open(url)
                    if hoang == 'Youtube':
                        inp3 = input("Bot Dz: Bạn muốn tìm kiếm gì ? \nYou: ")
                        url=f"https://www.youtube.com/search?q="+inp3
                        wb.get().open(url)
                    if hoang == 'Facebook':
                        url=f"https://www.facebook.com/"
                        wb.get().open(url)
                    if hoang == 'Ảnh':
                        link = r"C:\Users\PC\Pictures\poster.png"
                        os.startfile(link)
                        
            print("Bot Dz: "+random.choice(responses))
        else: 
            inp2 = input("Hãy nhập câu trả lời: ")
            update_json("training.json",inp,inp2)
            
    
chat()