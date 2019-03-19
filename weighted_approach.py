
import json
import pandas as pd
from pandas.io.json import json_normalize
from urllib import request
from bs4 import BeautifulSoup
import re
import datetime
import pandas as pd

def read_json(path):
    '''
    Reads json file from path and converts it into pandas dataframe
    :param path:
    :return dataframe:
    '''
    with open(path, 'r', encoding = 'utf') as f:
        train = json.load(f)
        train_df = json_normalize(train)
        return train_df





def calculate_count(df):
    '''
    Generates count dictionary for a key which has a list of values in the format [key : {value_1:5, value_2:10}]
    :param dataframe:
    :return dictionary:
    '''
    cuisine = {}
    for i in range(len(df)):
        each_cuisine = df[i:i+1]['cuisine'].values[0]
        if each_cuisine in cuisine.keys():
            ingredients_list = df[i:i+1]['ingredients'].values[0]
            for each_ing in ingredients_list:
                if each_ing in cuisine[each_cuisine].keys():
                    cuisine[each_cuisine][each_ing] += 1
                else:
                    cuisine[each_cuisine][each_ing] = 0
        else:
            ingredients_list = df[i:i + 1]['ingredients'].values[0]
            cuisine[each_cuisine] = {each_ing:0 for each_ing in ingredients_list}
    return cuisine





def calculate_weights(d):
    '''
    Generates weights for list in format [key : {value_1:w1, value_2:w2}]
    :param dictionary:
    :return count_list:
    '''
    for each_cuisine in d.keys():
        total_sum = sum(d[each_cuisine][k] for k in d[each_cuisine].keys())
        for each_ing in d[each_cuisine].keys():
            d[each_cuisine][each_ing] = round(d[each_cuisine][each_ing]/total_sum, 4)
    return d







def test_score(test_df):
    '''
    Predicting score for test dataframe based on train frame of the format [key : {value_1:w1, value_2:w2}], calculating weighted scores for all permutations
    :param dataframe:
    :return dataframe:
    '''
    # List for storing predictions
    pred = []
    # Iterating over rows of test
    for row in range(len(test_df)):
        # Creating an empty score_list
        score_list = []
        # Extracting ingredients from each test row
        ingredients_list = test_df[row:row + 1]['ingredients'].values[0]
        # Calculating score for each cuisine from extracted ingredients list
        for each_cuisine in weighted_cuisine_dict.keys():
            score = 0
            for each_ing in ingredients_list:
                # As ingredients are stored in the format {cuisine: {ind: score, ..}}, checking if ingredient is present or not for the cuisine
                if each_ing in weighted_cuisine_dict[each_cuisine].keys():
                    # If ingredient is then add the weight to score
                    score += weighted_cuisine_dict[each_cuisine][each_ing]
            # Score for each cuisine gets added to score_list for each row(ingredient list) in test
            score_list.append((each_cuisine, round(score, 4)))
        # Sorting score list based upon score
        score_list = sorted(score_list, key = lambda x:x[1], reverse = True)
        # For each row, appending the predicted cuisine match based upon sorted score_list
        pred.append(score_list[0][0])
    test_df['pred'] = pred
    return test_df

def list_to_seq(df_col):
    '''

    :param df_col:
    :return:
    '''
    return df_col.apply(lambda x : ' '.join(x))






if __name__=='__main__':
  train = read_json('/content/train.json')
  test = read_json('/content/test.json')
  train, val =  train_test_split(train, test_size=0.2, random_state=42)
  print(len(train), len(val))
  d = calculate_count(train)
  weighted_cuisine_dict = calculate_weights(d)
  val_features = val[['id', 'ingredients']]
  val_pred = test_score(val_features)
  print(classification_report(val_labels, val_pred['pred']))
  print(accuracy_score(val_labels, val_pred['pred']))

