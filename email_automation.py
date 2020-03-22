import mailparser

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D,Bidirectional,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk import word_tokenize
import pickle
from pprint import pprint
from nltk.util import ngrams
from pandas import DataFrame

def create_model():
    model = Sequential()
    model.add(Embedding(50000, 300, input_length=500))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.load_weights("/home/bid/Downloads/check_point")

with open ("/home/bid/Downloads/Pr_NoEmail2.pkl","rb")as handle:
  tokenizer = pickle.load(handle)

"""
Instructions:
First install deepavlov using command: pip install deeppavlov
Then use command : python -m deeppavlov install ner_ontonotes_bert 
""" 

from deeppavlov import build_model,configs
"""
Suggestion: For faster results create an object of the following "model" instead of creating one everytime  
"""
model_bert = build_model(configs.ner.ner_ontonotes_bert, download=True) #Just for the intial run keep the parameter download=True, for the subsequent runs remove the parameter download

"""Enter the filename of the mail file and comment the following 'text' for test run"""

"""
filename = "path to the email file"
mail = mailparser.parse_from_file(filename)
text = mail.text_plain
"""
 
"""For test run"""
text = 'Subject: Pre-proposal Conference - RFCSP #20-005A RG Campus:  Purchase and Installation of New Fire Alarm System Date: 2019-11-13 13:01 From: Jauregui, Mary M. <mlaird@epcc.edu> To: Undisclosed recipients:; Good afternoon. Please be reminded that the Pre-proposal Conference for the above referenced solicitation process will be held tomorrow, Thursday, November 14, 2019 beginning at 9:30 a.m. in Room A119 of the Rio Grande Campus, located at 100 West Rio Grande Avenue, El Paso, Texas 79902. The pre-proposal conference will be followed by a site walk-through of the area.  Please wear comfortable shoes and dress appropriately for the terrain and weather.https://weather.com/weather/monthly/l/USTX0413:1:US [1].'

text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
text = re.sub(r'http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
email_id = re.findall('\S+@\S+', text)

if email_id:
    for ids in email_id:
        text = text.replace(ids ," ")


entity = model_bert([text]) #For test run 'text'
#entity = model_bert(text) #For mail text

org = []
event = []
date = []
time = []
loc = []
gpe = []
wof = []
fac = []
bid = []

s_org = ''
s_event = ''
s_date = ''
s_loc = ''
s_gpe = ''
s_wof = ''
s_fac = ''
s_time = ''

for i in range(len(entity[0][0])):
    #print(entity[0][0][i],":",entity[1][0][i])
    if entity[1][0][i] == 'B-ORG':
        if s_org != '':
            org.append(s_org)
            s_org = ''
        else:
            s_org = entity[0][0][i]
    elif entity[1][0][i] == "I-ORG" or entity[1][0][i] == "L-ORG":
        s_org = s_org+" "+entity[0][0][i]
        
    elif entity[1][0][i] == 'B-EVENT':
        if s_event != '':
            event.append(s_event)
            s_event = ''
        else:
            s_event = entity[0][0][i]
    elif entity[1][0][i] == "I-EVENT" or entity[1][0][i] == "L-EVENT":
        s_event = s_event+ " " + entity[0][0][i]
        
    elif entity[1][0][i] == 'B-DATE':
        if s_date != '':
            date.append(s_date)
            s_date = ''
        else:
            s_date = entity[0][0][i]
        pd = entity[0][0][i]
    elif entity[1][0][i] == "I-DATE" or entity[0][0][i] == 'L-DATE':
        if pd == ':' or pd == '-' or entity[0][0][i] == ':' or entity[0][0][i] == '-' or entity[0][0][i] == ',':
            s_date = s_date + entity[0][0][i]
        else:
            s_date = s_date +" "+ entity[0][0][i]
        pd = entity[0][0][i]
        
    elif entity[1][0][i] == 'B-LOC':
        if s_loc != '':
            loc.append(s_loc)
            s_loc = ''
        else:
            s_loc = entity[0][0][i]
    elif entity[1][0][i] == "I-LOC" or entity[0][0][i] == 'L-LOC':
        s_loc = s_loc+ " "+entity[0][0][i]
        
    elif entity[1][0][i] == 'B-GPE':
        if s_gpe != '':
            gpe.append(s_gpe)
            s_gpe = ''
        else:
            s_gpe = entity[0][0][i]
    elif entity[1][0][i] == "I-GPE" or entity[0][0][i] == 'L-GPE':
        s_gpe = s_gpe+ " "+entity[0][0][i]
        
    elif entity[1][0][i] == 'B-WOF':
        if s_wof != '':
            wof.append(s_wof)
            s_wof = ''
        else:
            s_wof = entity[0][0][i]
    elif entity[1][0][i] == "I-WOF" or entity[0][0][i] == 'L-WOF':
        s_wof = s_wof+ " "+entity[0][0][i]
        
    elif entity[1][0][i] == 'B-FAC':
        if s_fac != '':
            fac.append(s_fac)
            s_fac = ''
        else:
            s_fac = entity[0][0][i]
    elif entity[1][0][i] == "I-FAC" or entity[0][0][i] == 'L-FAC':
        s_fac = s_fac+ " "+entity[0][0][i]
    
    elif entity[1][0][i] == 'B-TIME':
        if s_time != '':
            time.append(s_time)
            s_time = ''
        else:
            s_time = entity[0][0][i]
            pt = entity[0][0][i]
    elif entity[1][0][i] == "I-TIME" or entity[0][0][i] == 'L-TIME':
        if pt == ':' or pt == '-' or entity[0][0][i] == ':' or entity[0][0][i] == '-' or entity[0][0][i] == ',':
            s_time = s_time +entity[0][0][i]
        else:
            s_time = s_time +" "+ entity[0][0][i] 
        pt = entity[0][0][i]

if s_org != '':
    org.append(s_org)

if s_event != '':
    event.append(s_event)

if s_date != '':
    date.append(s_date)

if s_wof != '':
    wof.append(s_wof)
    
if s_fac != '':
    fac.append(s_fac)

if s_loc != '':
    loc.append(s_loc)
    
if s_gpe != '':
    gpe.append(s_gpe)

if s_time != '':
    time.append(s_time)
    
print("ORG:",org)
print("LOC:",loc)
print("GPE:",gpe)
print("EVENT:",event)
print("DATE:",date)
print("FAC:",fac)
print("WOF:",wof)
print("TIME:",time)

date_m = []
time_m = []

if date:
    for dates in date:
        text = text.replace(dates," ")
       
if time:
    for times in time:
        text = text.replace(times," ")
        
tk =Tokenizer(filters='!$%&*,;<=>?@[]^_`{|}~')


dictionary = {"Subject": []}
tokens = [token for token in text.split(" ")]
  #print(tokens)
  
X = list(ngrams(tokens,4))
Y = list(ngrams(tokens,3))
Z = list(ngrams(tokens,2))
A = list(ngrams(tokens,1))
  #print(X)
  #break

def get_grams(a_list):
    for items in X:
      string = ' '.join(x for x in items)
      dictionary["Subject"].append(string)

get_grams(X)
get_grams(Y)
get_grams(Z)
get_grams(A)
                
data = DataFrame(dictionary)
#print(data["Subject"].values)
tk.fit_on_texts(data["Subject"].values)       
#print(texts)
T = tk.texts_to_sequences(data["Subject"].values)
T = pad_sequences(T, maxlen = 500)
#print(T)
  #pprint(data.head())
predictions = model.predict(T)
  #pprint(predictions)
positive_predictions = [p[1] for p in predictions]
  #pprint(positive_predictions)
max_index = positive_predictions.index(max(positive_predictions))
#print(str(max(positive_predictions)),":",data.iloc[max_index,0])
dictionary={}
for p  in positive_predictions:
  index = positive_predictions.index(p)
  #print(str(p),":",data.iloc[index,0] )
  dictionary[data.iloc[index,0]] = p
  
S = (sorted(dictionary.items(),key = lambda kv:(float(kv[1]),(kv[0])),reverse = True))

#pprint(S)
#time.sleep(7)

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+')

bid_pred = []
for x in S:
  #print(x)
  #print(type(x))
  #print(x[0])
  word_tuple = x[0] 
  result = tokenizer.tokenize(word_tuple)
  flag = 0
  for word in result:
    for c in word:
      if c.isalpha() or c == '_':
        continue
      else:
        flag=1
        break

  if flag == 1:
    bid_pred.append(x)

bid_pred_counts = 0
for pred in bid_pred:
    bid.append(pred[0])
    bid_pred_counts+=1
    if bid_pred_counts == 10:
        break
print("Bid Number:",bid)
print("Email id:",email_id)
