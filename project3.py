# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from twython import Twython
import json
import pprint
import re
import nltk
from nltk import word_tokenize
from twython import Twython
from nltk.corpus import stopwords
from collections import defaultdict
from datetime import datetime, timedelta

apiKey = 'ISPrLuKZP3BhMjZ990qkKLA36'
apiSecret = 'feAaCueYAn0VYpqtK9YTu8RjCApsp25ABLwXjv3DTqcMKPvtGs'
bearerToken = 'AAAAAAAAAAAAAAAAAAAAADJiMwEAAAAASrtDxf6dOzIQpTd6m5kYbpb0aUI%3DbMcbJ078aEgDspPWybWjQOdZotz5aztxtIkmIHCR5hYfapmFgD'

accessToken = '1417014738-JWmhcvHtNUBoHVNORu9ptoilzH0KoAs5loypVBD'
accessTokenSecret = 'udzdwTP1ljEbD0HJwIFzkUIBCpq0VVSCx9ewJxoyWptvL'

twitter = Twython(app_key=apiKey, app_secret=apiSecret)

statuses = twitter.search(q='"Ted Cruz", lang:en', tweet_mode='extended', count=5000)['statuses']
for status in statuses:
    if 'retweeted_status' in status:
        status['full_text'] = status["retweeted_status"]['full_text']


#%%
texts = [status['full_text'] for status in statuses]

# Legacy cards
cruz = [text.lower().count('cruz') for text in texts]
cancun = [text.lower().count('cancun') for text in texts]
texas = [text.lower().count('texas') for text in texts]
winter = [text.lower().count('winter') for text in texts]
dhash = {'Cruz':cruz, 'Cancun':cancun, 'Texas':texas, 'winter': winter}
#df = pd.DataFrame(data=d)

#df.head(50)
# search = twitter.search(q='"Danaerys", #GameOfThrones, since:2011-04-17', tweet_mode='extended', count=5000)
# search = twitter.search(q='"Dani", love OR hate, #GameOfThrones, lang:en, until:2021-01-01, since:2011-04-17, -filter:links, -filter:replies')

# print(search)

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


def getpopwords(searchterm):
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
            text = status['full_text']
            text = re.sub(r'[^\w]', ' ', text)
            text = word_tokenize(text)
            text = [word for word in text if not word in stopwords.words()]
            for word in text:
                dwords[word] += 1

        # gets the next set of statuses
        statuses = twitter.search(q=searchterm, tweet_mode='extended', lang='en', count=100, max_id=newid)['statuses']
        newid = statuses[len(statuses) - 1]['id']
        totalapirequests += 1
        print("api requests:", totalapirequests)
        if totalapirequests > 50:
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
# getpopwords('cruz')
#
# lists = sorted(dwords.items(), key=lambda item: item[1], reverse=True)
# x, y = zip(*lists)  # unpack a list of pairs into two tuples
# x = x[:10]
# y = y[:10]
#
# # display most popular 10 tweets over timeframe starting 6 days ago and ending when
# # i run out of api requests
# # 900 / 15 min or 100,000 for a day
# plt.plot(x, y)
# plt.show()

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
