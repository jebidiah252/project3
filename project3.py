# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from twython import Twython

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
d = {'Cruz':cruz, 'Cancun':cancun, 'Texas':texas, 'winter': winter}
df = pd.DataFrame(data=d)

df.head(50)
# search = twitter.search(q='"Danaerys", #GameOfThrones, since:2011-04-17', tweet_mode='extended', count=5000)
# search = twitter.search(q='"Dani", love OR hate, #GameOfThrones, lang:en, until:2021-01-01, since:2011-04-17, -filter:links, -filter:replies')

# print(search)