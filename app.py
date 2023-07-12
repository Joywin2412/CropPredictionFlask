from flask import Flask , request,jsonify
import numpy as np
import pandas as pd
import random
from textblob import TextBlob
from joblib import load
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras_preprocessing.sequence import pad_sequences


app = Flask(__name__)
model = load('xgboostmodel.joblib')
chatmodel = load('chatbotmodel.joblib')
tokenizer_obj = load('tokenizer.joblib')

def clean_text(text):
    all_reviews = []
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    now_text = text[0]
    now_text = now_text.lower()
    now_text = pattern.sub('', now_text)
    now_text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", now_text)
    now_text = re.sub(r"i'm", "i am", now_text)
    now_text = re.sub(r"he's", "he is", now_text)
    now_text = re.sub(r"she's", "she is", now_text)
    now_text = re.sub(r"that's", "that is", now_text)        
    now_text = re.sub(r"what's", "what is", now_text)
    now_text = re.sub(r"where's", "where is", now_text) 
    now_text = re.sub(r"\'ll", " will", now_text)  
    now_text = re.sub(r"\'ve", " have"  , now_text)  
    now_text = re.sub(r"\'re", " are", now_text)
    now_text = re.sub(r"\'d", " would", now_text)
    now_text = re.sub(r"\'ve", " have", now_text)
    now_text = re.sub(r"won't", "will not", now_text)
    now_text = re.sub(r"don't", "do not", now_text)
    now_text = re.sub(r"did't", "did not", now_text)
    now_text = re.sub(r"can't", "can not", now_text)
    now_text = re.sub(r"it's", "it is", now_text)
    now_text = re.sub(r"couldn't", "could not", now_text)
    now_text = re.sub(r"have't", "have not", now_text)
    tokens = word_tokenize(now_text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]

    PS = PorterStemmer()
    words = [PS.stem(w) for w in words ]
    words = ' '.join(words)
    all_reviews.append(words)
    return all_reviews




@app.route("/",methods = ['POST'])
def prediction():
    data = request.get_json(force=True)
    state_name = data['state']
    season = data['season']
    crop = data['crop']
    area = float(data['area'])
    df2 = pd.DataFrame(0,index = range(1), columns = ['Area',
 'State_Name_AndamanandNicobarIslands',
 'State_Name_AndhraPradesh',
 'State_Name_ArunachalPradesh',
 'State_Name_Assam',
 'State_Name_Bihar',
 'State_Name_Chandigarh',
 'State_Name_Chhattisgarh',
 'State_Name_DadraandNagarHaveli',
 'State_Name_Goa',
 'State_Name_Gujarat',
 'State_Name_Haryana',
 'State_Name_HimachalPradesh',
 'State_Name_JammuandKashmir',
 'State_Name_Jharkhand',
 'State_Name_Karnataka',
 'State_Name_Kerala',
 'State_Name_MadhyaPradesh',
 'State_Name_Maharashtra',
 'State_Name_Manipur',
 'State_Name_Meghalaya',
 'State_Name_Mizoram',
 'State_Name_Nagaland',
 'State_Name_Odisha',
 'State_Name_Puducherry',
 'State_Name_Punjab',
 'State_Name_Rajasthan',
 'State_Name_Sikkim',
 'State_Name_TamilNadu',
 'State_Name_Telangana',
 'State_Name_Tripura',
 'State_Name_UttarPradesh',
 'State_Name_Uttarakhand',
 'State_Name_WestBengal',
 'Season_Autumn',
 'Season_Kharif',
 'Season_Rabi',
 'Season_Summer',
 'Season_WholeYear',
 'Season_Winter',
 'Crop_Apple',
 'Crop_Arcanut(Processed)',
 'Crop_Arecanut',
 'Crop_Arhar/Tur',
 'Crop_AshGourd',
 'Crop_Atcanut(Raw)',
 'Crop_Bajra',
 'Crop_Banana',
 'Crop_Barley',
 'Crop_Bean',
 'Crop_Beans&Mutter(Vegetable)',
 'Crop_BeetRoot',
 'Crop_Ber',
 'Crop_Bhindi',
 'Crop_BitterGourd',
 'Crop_Blackgram',
 'Crop_Blackpepper',
 'Crop_BottleGourd',
 'Crop_Brinjal',
 'Crop_Cabbage',
 'Crop_Cardamom',
 'Crop_Carrot',
 'Crop_Cashewnut',
 'Crop_CashewnutProcessed',
 'Crop_CashewnutRaw',
 'Crop_Castorseed',
 'Crop_Cauliflower',
 'Crop_CitrusFruit',
 'Crop_Coconut',
 'Crop_Coffee',
 'Crop_Colocosia',
 'Crop_Cond-spcsother',
 'Crop_Coriander',
 'Crop_Cotton(lint)',
 'Crop_Cowpea(Lobia)',
 'Crop_Cucumber',
 'Crop_DrumStick',
 'Crop_Drychillies',
 'Crop_Dryginger',
 'Crop_Garlic',
 'Crop_Ginger',
 'Crop_Gram',
 'Crop_Grapes',
 'Crop_Groundnut',
 'Crop_Guarseed',
 'Crop_Horse-gram',
 'Crop_JackFruit',
 'Crop_Jobster',
 'Crop_Jowar',
 'Crop_Jute',
 'Crop_Jute&mesta',
 'Crop_Kapas',
 'Crop_Khesari',
 'Crop_Korra',
 'Crop_Lab-Lab',
 'Crop_Lemon',
 'Crop_Lentil',
 'Crop_Linseed',
 'Crop_Litchi',
 'Crop_Maize',
 'Crop_Mango',
 'Crop_Masoor',
 'Crop_Mesta',
 'Crop_Moong(GreenGram)',
 'Crop_Moth',
 'Crop_Nigerseed',
 'Crop_Oilseedstotal',
 'Crop_Onion',
 'Crop_Orange',
 'Crop_OtherCereals&Millets',
 'Crop_OtherCitrusFruit',
 'Crop_OtherDryFruit',
 'Crop_OtherFreshFruits',
 'Crop_OtherKharifpulses',
 'Crop_OtherRabipulses',
 'Crop_OtherVegetables',
 'Crop_Paddy',
 'Crop_Papaya',
 'Crop_Peach',
 'Crop_Pear',
 'Crop_Peas&beans(Pulses)',
 'Crop_Peas(vegetable)',
 'Crop_Perilla',
 'Crop_Pineapple',
 'Crop_Plums',
 'Crop_PomeFruit',
 'Crop_PomeGranet',
 'Crop_Potato',
 'Crop_Pulsestotal',
 'Crop_PumpKin',
 'Crop_Ragi',
 'Crop_RajmashKholar',
 'Crop_Rapeseed&Mustard',
 'Crop_Redish',
 'Crop_RibedGuard',
 'Crop_Rice',
 'Crop_Ricebean(nagadal)',
 'Crop_Rubber',
 'Crop_Safflower',
 'Crop_Samai',
 'Crop_Sannhamp',
 'Crop_Sapota',
 'Crop_Sesamum',
 'Crop_Smallmillets',
 'Crop_SnakGuard',
 'Crop_Soyabean',
 'Crop_Sugarcane',
 'Crop_Sunflower',
 'Crop_Sweetpotato',
 'Crop_Tapioca',
 'Crop_Tea',
 'Crop_Tobacco',
 'Crop_Tomato',
 'Crop_Totalfoodgrain',
 'Crop_Turmeric',
 'Crop_Turnip',
 'Crop_Urad',
 'Crop_Varagu',
 'Crop_WaterMelon',
 'Crop_Wheat',
 'Crop_Yam',
 'Crop_otherfibres',
 'Crop_othermisc.pulses',
 'Crop_otheroilseeds'])
    df2['State_Name_'+state_name] = 1
    df2['Season_'+season] = 1
    df2['Crop_'+crop] = 1
    df2['Area'] = area
    print(df2.shape)
    Production = model.predict(df2)
    return (str(Production*area))

@app.route("/chat",methods = ['POST'])
def chat():
    data = request.get_json(force=True)
    text = data['text']
    dict = {'predict': 0,
 'recommend': 1,
 'goodbye': 2,
 'name': 3,
 'greeting': 4,
 'weather': 5,
 'stores': 6,
 'crops': 7,
 'friends': 8,
 'delete': 9,
 'notfound': 10,
 'communication': 11}
    dictionary2 = {'predict': ['You can go to our homepage and click on get started 🤝'],
 'recommend': ['Congrats! You accessed one of to-be-developed features. I am sincerely sorry to tell that the feature is not yet available',
  'Features is in development stage. Inconvience regretted'],
 'goodbye': ['Bye', 'take care'],
 'name': ['My name is farm sensei. Developers named me that',
  'I am farm sensei. Ready to be at your service'],
 'greeting': ['Hi there. I am Farm Sensei!. Ready to help', 'Hello', 'Hi :)'],
 'weather': ['Go to homepage and access weather forecast for our amazing features.'],
 'stores': ['Farm sensei to the rescue! You can access stores from the homepage!',
  'To know the stores near you can use our stores feature in the homepage'],
 'crops': ['Farm sensei to the rescue! You can access crops from the major crops in the homepage!',
  'To know the crops grown near you can use our major crops feature in the homepage 🚀'],
 'friends': ['Farm sensei to the rescue! Click on your profile on the toppage! There you go you should see a friends bar on the right!!'],
 'delete': ['Extremely sorry to hear that! We will improve in the future. If you are unhappy with our service delete the account is present in the profile accessed by clicking on the top of your page'],
 'notfound': ['Farm Sensei encourage to fill the form in the profile page. Profile is accessed by clicking on the top of the page. Then you must be able to see the farmers near you!!'],
 'communication': ['Farm Sensei encourage to go to profile page. Then you must be able to see the farmers near your area. Send them a friend request. And after accepting you will be able to see their phone number']}
    text = str(TextBlob(text).correct())
    print(text)
    test_lines = clean_text([text])
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    print(test_sequences)
    # consider = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    test_review_pad = pad_sequences(test_sequences, maxlen=15, padding='post')
    print(test_review_pad)
    
    pred = chatmodel.predict([test_review_pad])
    pred*=100
    pred[0] = np.array(pred[0])
    print(pred)
    i = np.argmax(pred[0])
    inverse_dict = {value: key for key, value in dict.items()}
    print(inverse_dict[i])
    ourResult = random.choice(dictionary2[inverse_dict[i]])
    return (ourResult)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

