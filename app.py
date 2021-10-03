from flask import Flask, render_template, request
from flask import session
import uuid
import random
import io
import re
import time
from datetime import datetime
import numpy as np
import pickle
import os
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
import copy
import transformers
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import language_tool_python
from string import punctuation

cwd = os.getcwd()

pickle_path = cwd + "/chatlogs/"

# pickled models
# concern identification models and feature representation
filename = 'finalized_model.sav'
filename1 = 'finalized_model_agreement.sav'
filename2 = "transformer.sav"
filename3 = "transformer_agreement.sav"
# question identification model
filename4 = 'finalized_model_q.sav'
filename5 = "transformer_q.sav"
# answer identification model
filename6 = 'question_model.sav'
                       
                        
try:
    # load the model from disk
    concern_model = pickle.load(open(filename, 'rb'))
    concern_model_agreement = pickle.load(open(filename1, 'rb'))
    transformer = pickle.load(open(filename2, 'rb'))
    transformer_agreement = pickle.load(open(filename3, 'rb'))
    answer_model = pickle.load(open(filename4, 'rb'))
    answer_token = pickle.load(open(filename5, 'rb'))


except:
    #download models from google drive
    pip install gdown
    gdown https://drive.google.com/uc?id=1aPDfqR_siXbaeJlWA_bEgq83tnlywoDc
    gdown https://drive.google.com/uc?id=1CUHKIV-wz0_YLDt3XSXkCe93nLa56xRH
    gdown https://drive.google.com/uc?id=1rIVycihBjaTSI1GqqTG5-3pKuc5QD5Fm
    gdown https://drive.google.com/uc?id=1cik-I0u5Hi1n-QJcHhmz04s-z46yKrmF
    gdown https://drive.google.com/uc?id=1ePDQU6ptb-oLrjjBCUilxMAIgHHqj3tE
    gdown https://drive.google.com/uc?id=1mOR_c9DAXavs6CVpAtXLLNd-hCpqHxpb
    gdown https://drive.google.com/uc?id=1Q3t_0VmX39iXvU5hsYBpIdm0VaHq0omY

    # load the model from disk
    concern_model = pickle.load(open(filename, 'rb'))
    concern_model_agreement = pickle.load(open(filename1, 'rb'))
    transformer = pickle.load(open(filename2, 'rb'))
    transformer_agreement = pickle.load(open(filename3, 'rb'))
    answer_model = pickle.load(open(filename4, 'rb'))
    answer_token = pickle.load(open(filename5, 'rb'))

stop_words_file = 'SmartStoplist.txt'

stop_words = []

with open(stop_words_file, "r") as f:
    for line in f:
        stop_words.extend(line.split())

with open('concern_dic.pickle', 'rb') as handle:
    concern_dic = pickle.load(handle)

with open('id_dic.pickle', 'rb') as handle:
    id_dic = pickle.load(handle)

with open('q_concern_dic.pickle', 'rb') as handle:
    q_concern_dic = pickle.load(handle)

with open('q_id_dic.pickle', 'rb') as handle:
    q_id_dic = pickle.load(handle)
    
# unpickle BERT model for machines without GPU    
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# BERT question sentence classifier        
bert_classifier = CPU_Unpickler(open(filename6, 'rb')).load()    
    

    
# preprocess user input    
def preprocess(raw_text):


    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = word_tokenize(letters_only_text.lower())

    # remove stopwords
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)

    lemmatised_words = []
    lemmatizer = WordNetLemmatizer()
    for word in cleaned_words:
        word = lemmatizer.lemmatize(word)
        lemmatised_words.append(word)

    return lemmatised_words

# return most likely categories of concern for each input
def get_top_k_predictions_(model,X_test,k):

    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    prob_list = list(probs[0])
    prob_list.sort(reverse=True)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]

    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds=[ item[::-1] for item in preds]
    preds = preds[0]

    pred_prob = prob_list[:k]

    return preds, pred_prob


# returns argument according to the related concern
def return_arg_and_concern(user_mes, concern_dic ): #, prev_cb_responses):

    response_id = 0
    user_mes = user_mes.lower()

    # checking first whether person agrees - then no need to check for counterarg

    disagree = ['dont', "don't", "not", 'lie', 'disagree', 'no']


    bool_disagree = any(x in user_mes.split() for x in disagree)



    # if user agrees we use a default argument
    if bool_disagree == True and len(user_mes.split()) < 7:
        print('disagreement.')
        concern = 'disagree'
        possible_responses = concern_dic['default']
        return (concern, response_id)



    # now lets preprocess and look for a match in the KB
    message_prep = preprocess(user_mes)
    message_sen = sen = ' '.join(message_prep) # as string for classifier
    message_features = transformer_agreement.transform([message_sen])
    concerns_, preds = get_top_k_predictions_(concern_model_agreement, message_features, 2)
    print(concerns_, preds)
    #print(concerns_)

    """  check if agreement   """
    if concerns_[0] == 'agree' and preds[0] > 0.5 and len(user_mes.split()) < 13:
        concern = 'agree'

        possible_responses = concern_dic['default']
        if possible_responses == []:
            concern = "no concern"
            return (concern, response_id)
        else:
            response_id = possible_responses[0]
            return (concern, response_id)

    message_features = transformer.transform([message_sen])
    concerns_, preds = get_top_k_predictions_(concern_model, message_features, 2)
    #print(concerns_, preds)
    print(concerns_)
    """
    NO CONCERN - SO RETURN DEFAULT ARGUMENT OR IF THATS EMPTY RESPONSE ID = 0
    """
    if preds[0] < 0.4:
        concern = 'default'
        possible_responses = concern_dic[concern]
        if possible_responses == []:
            concern = "no concern"
            return (concern, response_id)
        else:
            response_id = possible_responses[0]
            return (concern, response_id)


    else:
        concern = concerns_[0]

        possible_responses_1 = concern_dic[concern]
        possible_responses_2 = concern_dic['default']

        if possible_responses_1 == []:
            if possible_responses_2 != []:
                response_id = possible_responses_2[0]
                return('default', response_id)
            else:
                concern = "no concern"
                return (concern, response_id)
        else:
            response_id = possible_responses_1[0]
            return (concern, response_id)


        
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

# identifies inputs as questions or not
def is_question(user_input):
    
    # encode input strings
    encoded_data_vals = tokenizer.encode_plus(
        user_input, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=20, 
        return_tensors='pt'
    )


    input_ids_vals = encoded_data_vals['input_ids']
    attention_masks_vals = encoded_data_vals['attention_mask']
    dataset_vals = TensorDataset(input_ids_vals, attention_masks_vals)

    dataloader_validations = DataLoader(dataset_vals, 
                                       sampler=SequentialSampler(dataset_vals)) 
    
    
    for batch in dataloader_validations:
        
       
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],                  
                 }

        
        outputs = bert_classifier(**inputs)
            
            
            
        logits = outputs[0]
        
        if logits.data[0][0] > 0:
            return False
        else:
            return True

    

# correct spelling and punctuation errors in chatbot output
def grammar_correct(text):   
    tool = language_tool_python.LanguageTool('en-US')  
    matches = tool.check(text)

    my_mistakes = []
    my_corrections = []
    start_positions = []
    end_positions = []
    for rules in matches:
        if len(rules.replacements)>0:
            start_positions.append(rules.offset)
            end_positions.append(rules.errorLength+rules.offset)
            my_mistakes.append(text[rules.offset:rules.errorLength+rules.offset])
            my_corrections.append(rules.replacements[0])

    my_new_text = list(text)

    for m in range(len(start_positions)):
        for i in range(len(text)):
            my_new_text[start_positions[m]] = my_corrections[m]
            if (i>start_positions[m] and i<end_positions[m]):
                my_new_text[i]=""

    my_new_text = "".join(my_new_text)
    my_new_text = my_new_text.lstrip()
    my_new_text = my_new_text.rstrip(punctuation)
    correct_text = " " + my_new_text[0].lower() + my_new_text[1:]
    
    if correct_text.endswith('.'):
        return correct_text
    else:
        return correct_text + '.'    
    

# returns likelihood scores for the first and last word of the answer to user question     
def question_entry(quest, concern):

    question = f''' {quest} '''

    vaccine_info = q_id_dic[f'{concern}']

    paragraph = f''' {vaccine_info} '''

    encoding = answer_token.encode_plus(text=question,text_pair=paragraph, add_special=True)

    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = answer_token.convert_ids_to_tokens(inputs) #input tokens
    start_scores, end_scores = answer_model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    
    return start_scores, end_scores, tokens
  
# use predicted values to return an answer     
def get_answer(start_scores, end_scores, tokens):
    
    # returns index of first and last word of best answer
    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    # indexes and prob of top 8 most likely end words for answer
    predicted_k_indexes = torch.topk(end_scores,k=8)
    prk_0 = predicted_k_indexes[0]
    prk_1 = predicted_k_indexes[1]

    # indexes and prob of top 8 most likely start words for answer
    s_predicted_k_indexes = torch.topk(start_scores,k=8)
    s_prk_0 = s_predicted_k_indexes[0]
    s_prk_1 = s_predicted_k_indexes[1]
    
    # if prob of end word is less than 2 return argument rather than anwer
    if prk_0[0][0] < 2:
        return None
    
    # if more descriptive answers (incorporating more words) are close in likelihood return the longer answer
    i = 0
    best_end_pr = prk_0[0][0]
    best_end_pos = prk_1[0][0]    
    for n in range(5):
        if (2 * prk_0[0][i+1]) > best_end_pr and prk_1[0][i+1] > best_end_pos and not prk_1[0][i+1] > best_end_pos + 40:
            best_end_pr = prk_0[0][i+1]
            best_end_pos = prk_1[0][i+1]
            end_index = best_end_pos
            i = i+1
        elif (2 * prk_0[0][i+1]) > best_end_pr and not (prk_1[0][i+1] > best_end_pos and not prk_1[0][i+1] > best_end_pos + 40):                                          
            i = i+1


    j = 0
    best_start_pr = s_prk_0[0][0]
    best_start_pos = s_prk_1[0][0]
    for n in range(5):
        if (1.5 * s_prk_0[0][j+1]) > best_start_pr and s_prk_1[0][j+1] < best_start_pos and not s_prk_1[0][j+1] < best_start_pos - 15:  
            best_start_pr = s_prk_0[0][j+1]
            best_start_pos = s_prk_1[0][j+1]        
            start_index = s_prk_1[0][j+1]
            j = j+1
        elif (1.5 * s_prk_0[0][j+1]) > best_start_pr and not s_prk_1[0][j+1] < best_start_pos and not s_prk_1[0][j+1] < best_start_pos - 15:                                             
            j = j+1       
            

    answer = ' '.join(tokens[start_index:end_index+1])
    
    corrected_answer = ''

    for word in answer.split():

        # if it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer





  

    
    
    
    

    
    

app = Flask(__name__)
app.static_folder = 'static'



app.secret_key = "coronavirus"

@app.before_request
def make_session_permanent():
    session.permanent = False

@app.route("/")
def home():
    session.clear()
    print(session)
    if "user_id" not in session:
        print('new user')
        # use this session["user_id"], try printing
        user_id = str(uuid.uuid1())
        session["user_id"] = uuid.uuid1()
        session['concern_dic'] = copy.deepcopy(concern_dic)
        session['q_concern_dic'] = copy.deepcopy(q_concern_dic)        
        session['chatlogs'] = []



    return render_template("index.html", user_id=session["user_id"])





@app.route("/get")
def get_bot_response():

    user_mes = request.args.get('msg')
    stop = user_mes.lower()

    if "prolific_id" not in session:
        session['prolific_id'] = user_mes
        bot_reply_1 = "Nice to meet you! You can always let me know you want to end the chat by typing 'quit'.\n Throughout our chat I might post some links with extra information about a particular topic we've discussed. If you want to check them out, please copy-paste them into a new browser window and do not close this one. Ok?"
        session['chatlogs'].append(user_mes)
        #print(len(session['chatlogs']))
        return bot_reply_1

    elif len(session['chatlogs']) == 1:
        bot_reply_2 = "Cool! Please ask me any questions you have about the COVID-19 vaccine, or let me know some reasons why you don't want to take one (once one becomes available to you)."
        session['chatlogs'].append("START")
        return bot_reply_2

    elif stop == 'quit':
        bot_reply = "You are ending the chat. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine if it becomes available. Good bye! "

        log_mes= "User: " + user_mes
        session['chatlogs'].append(log_mes)
        session['chatlogs'].append("END")

        pickle_file_name = pickle_path + str(session["user_id"]) + ".pickle"

        with open(pickle_file_name, 'wb') as handle:
            pickle.dump(session["chatlogs"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        #time.sleep(random.randint(5, 10))
        return bot_reply

    else:
    

        if is_question(user_mes) == True:
            
            while True:
                concern, response_id = return_arg_and_concern(user_mes, session['q_concern_dic'])
#                 if concern.startswith('default'):
#                     break

                
                
                q_con_dic_user = session['q_concern_dic']
                s_scores, e_scores, toks = question_entry(user_mes, concern)
                bot_reply = get_answer(s_scores, e_scores, toks)
                add_question = ["Good question, the answer is", 'I believe the answer to that is', 'Thanks for asking, the answer is', 'I\'m glad you asked, the answer to that is']
                ag = random.choice(add_question)
                try:
                    bot_reply = ag + grammar_correct(bot_reply)
                except:
                    break

#                 q_con_dic_user[concern] = q_con_dic_user[concern][1:]
#                 session['q_concern_dic'] = q_con_dic_user


                chatbot_response = "CB: " + str(response_id)
                session['chatlogs'].append(chatbot_response)
        
                if bot_reply is not None or not " ":            
                    return bot_reply
                else:
                    break
        
        
        concern, response_id = return_arg_and_concern(user_mes, session['concern_dic'])

        if session['chatlogs'][-1] == "END":
            bot_reply = "Good bye :)"
            return bot_reply

        if concern == 'disagree':
            disag = ['Why?', 'Why not?']
            bot_reply = random.choice(disag)
            #time.sleep(random.randint(3, 5))
            return bot_reply

        if response_id == 0:
            bot_reply = "I'll stop here. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine. Good bye! "

            #adding last argument where no match was found to chatlog
            log_mes= "User: " + user_mes
            session['chatlogs'].append(log_mes)
            session['chatlogs'].append("END")

            pickle_file_name = pickle_path + str(session["user_id"]) + ".pickle"
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(session["chatlogs"], handle, protocol=pickle.HIGHEST_PROTOCOL)
            #time.sleep(random.randint(5, 10))
            return bot_reply

        else:
            add_a = False
            add_d = False
            add_default = ["Noted. But ", 'I understand, however, ', 'But have you considered that ', 'Nevertheless, ', 'Nonetheless, ', 'Despite that, ']
            add_agree = ['Thanks. Also, have you considered that ', 'I\'m glad. Also, ', 'I\'m happy you agree. Don\'t you also think that ']
            #check whether agree or DEFAULT
            print(concern)
            if concern == 'default':
                add_d = True

            if concern == 'agree':
                add_a = True
                concern = 'default'



            log_mes= "User: " + user_mes
            session['chatlogs'].append(log_mes)
            #retrieving the concern dictionary for the user

            con_dic_user = session['concern_dic']
            bot_reply = id_dic[response_id][0]

            con_dic_user[concern] = con_dic_user[concern][1:]
            session['concern_dic'] = con_dic_user


            chatbot_response = "CB: " + str(response_id)
            session['chatlogs'].append(chatbot_response)

            if add_a == True:
                ag = random.choice(add_agree)
                bot_reply = ag + bot_reply
            if add_d == True:
                ag = random.choice(add_default)
                bot_reply = ag + bot_reply

            #time.sleep(random.randint(5, 10))
            return bot_reply




    return bot_reply

if __name__ == "__main__":
    app.run(debug=True)
