# %%
import json
import pprint
import re
import string
import random

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from twython import Twython

from collections import defaultdict
from datetime import datetime, timedelta

stop_words = stopwords.words('english')

# This code is from the following link
# https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
# this link helped me to figure out the best way to perform sentiment analysis on
# the Ted Cruz twitter data.

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

apiKey = 'ISPrLuKZP3BhMjZ990qkKLA36'
apiSecret = 'feAaCueYAn0VYpqtK9YTu8RjCApsp25ABLwXjv3DTqcMKPvtGs'
bearerToken = 'AAAAAAAAAAAAAAAAAAAAADJiMwEAAAAASrtDxf6dOzIQpTd6m5kYbpb0aUI%3DbMcbJ078aEgDspPWybWjQOdZotz5aztxtIkmIHCR5hYfapmFgD'

accessToken = '1417014738-JWmhcvHtNUBoHVNORu9ptoilzH0KoAs5loypVBD'
accessTokenSecret = 'udzdwTP1ljEbD0HJwIFzkUIBCpq0VVSCx9ewJxoyWptvL'

twitter = Twython(app_key=apiKey, app_secret=apiSecret)

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

stop_words = stopwords.words('english')

pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

pos_cleaned_tokens = []
neg_cleaned_tokens = []

for tokens in pos_tweet_tokens:
    pos_cleaned_tokens.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(pos_cleaned_tokens)
pos_tokens_for_model = get_tweets_for_model(pos_cleaned_tokens)

pos_dataset = [(tweet_dict, 'Positive') for tweet_dict in pos_tokens_for_model]

# freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))

for tokens in neg_tweet_tokens:
    neg_cleaned_tokens.append(remove_noise(tokens, stop_words))

all_neg_words = get_all_words(neg_cleaned_tokens)
neg_tokens_for_model = get_tweets_for_model(neg_cleaned_tokens)

neg_dataset = [(tweet_dict, 'Negative') for tweet_dict in neg_tokens_for_model]

dataset = pos_dataset + neg_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

print('Accuracy is:', classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

# %%

# search = twitter.search(q='Ted Cruz', result_type='extended', verified='True', filter_out='retweets')['statuses']

# for i in search:
#     if 'user' in i:
#         print(i['user'])
#         break

#%%
dhash = defaultdict(int)

def getdaysago(days):
    return (datetime.today() - timedelta(days)).strftime('%Y-%m-%d')


def gethashtags(searchterm):
    statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, until=getdaysago(6))[
        'statuses']
    newid = statuses[len(statuses) - 1]['id']
    endofweekid = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=1)['statuses'][0]['id']

    # this will get a tally of how many times each unique hashtag was used
    totalapirequests = 0
    while newid < endofweekid:
        print(statuses[0]['created_at'])

        # fix retweets to not show username
        for status in statuses:
            if 'retweeted_status' in status:
                status['full_text'] = status["retweeted_status"]['full_text']

        # increment tweets in dicto based on num of appearances
        for status in statuses:
            if len(status['entities']['hashtags']) > 0:
                for item in status['entities']['hashtags']:
                    dhash[item['text']] += 1

        # gets the next set of statuses
        statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, max_id=newid)['statuses']
        newid = statuses[len(statuses) - 1]['id']
        totalapirequests += 1
        print("api requests:", totalapirequests)
        if totalapirequests > 50:
            break

dwords = defaultdict(int)
dposneg = defaultdict(int)

def getpopwords(searchterm):
    statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, until=getdaysago(6))['statuses']
    newid = statuses[len(statuses) - 1]['id']
    endofweekid = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=1)['statuses'][0]['id']

    # this will get a tally of how many times each unique hashtag was used
    totalapirequests = 0
    while newid < endofweekid:
        # fix retweets to not show username
        for status in statuses:
            if 'retweeted_status' in status:
                status['full_text'] = status["retweeted_status"]['full_text']

        # increment tweets in dicto based on num of appearances
        has_been_accounted_for = []
        for status in statuses:
            text = status['full_text']
            text = re.sub(r'[^\w]', ' ', text)
            text = remove_noise(word_tokenize(status['full_text']), stop_words)
            if text in has_been_accounted_for:
                continue
            has_been_accounted_for.append(text)
            dposneg[classifier.classify(dict([token, True] for token in text))] += 1
            for word in text:
                dwords[word] += 1

        # gets the next set of statuses
        statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, max_id=newid)['statuses']
        newid = statuses[len(statuses) - 1]['id']
        totalapirequests += 1
        if dposneg['Positive'] + dposneg['Negative'] >= 5000:
            break


def gethashtagsbyday(searchterm):
    # this will get a tally of how many times each unique hashtag was used
    dlist = []
    endofweekid = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=1)['statuses'][0]['id']
    for i in range(6, 0, -1):
        dhashbyday = defaultdict(int)
        # increment day
        newday = getdaysago(i)
        statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, until=newday)['statuses']
        totalapirequests = 0
        newid = statuses[len(statuses) - 1]['id']
        while newid < endofweekid:
            statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, max_id=newid)[
                'statuses']
            newid = statuses[len(statuses) - 1]['id']
            print(statuses[0]['created_at'])

            # fix retweets to not show username
            for status in statuses:
                if 'retweeted_status' in status:
                    status['full_text'] = status["retweeted_status"]['full_text']

            # increment tweets in dicto based on num of appearances
            for status in statuses:
                if len(status['entities']['hashtags']) > 0:
                    for item in status['entities']['hashtags']:
                        dhashbyday[item['text']] += 1
            totalapirequests += 1
            if totalapirequests == 10:
                break
        dlist.append((newday[5:], dhashbyday))
    return dlist


# # get most popular hashtags by tweets including search term
# gethashtags('cruz')
#
# lists = sorted(dhash.items(), key=lambda item: item[1], reverse=True)
# x, y = zip(*lists)  # unpack a list of pairs into two tuples
# x = x[:10]
# y = y[:10]
#
# # display most popular 10 tweets over timeframe starting 6 days ago and ending when
# # i run out of api requests
# # 900 / 15 min or 100,000 for a day
# plt.plot(x, y)
# plt.show()

##############################################################

# # get most popular words in tweet by tweets including search term
getpopwords('Ted Cruz')

del dwords['ted']
del dwords['cruz']
del dwords['http']
del dwords['\'s']
del dwords['’']

# del dwords['"']
# del dwords['`']

lists = sorted(dwords.items(), key=lambda item: item[1], reverse=True)
x, y = zip(*lists)  # unpack a list of pairs into two tuples
x = x[:10]
y = y[:10]
#
# # display most popular 10 tweets over timeframe starting 6 days ago and ending when
# # i run out of api requests
# # 900 / 15 min or 100,000 for a day
plt.plot(x, y)
plt.show()

# %%

d = { 'Positive':[dposneg['Positive']], 'Negative':[dposneg['Negative']] }

df = pd.DataFrame(data=d)

sns.displot(data=df)
plt.figure()
plt.show()


# %%
# get most popular words in tweet by tweets including search term
temp = gethashtagsbyday('cruz')

megalist = []
for day in temp:
    megalist.append((day[0], (sorted(day[1].items(), key=lambda item: item[1], reverse=True)[0])))

x = [str(i[0]) + ' ' + str(i[1][0]) for i in megalist]
y = [i[1][1] for i in megalist]

# display most popular 10 tweets over timeframe starting 6 days ago and ending when
# i run out of api requests
# 900 / 15 min or 100,000 for a day
plt.plot(x, y)
plt.show()
