
# coding: utf-8

# In[1]:

import os
import sys
import json
from textblob import TextBlob
import matplotlib.pyplot as plot
import numpy as np

# Variables used 
tweets = []

jsonFiles = []
aa = ''
pola = []
subj = []

#Defining function to remove tweets with non-ascii characters
def remove_non_ascii(string):
    return ''.join(i if ord(i) <127 else ' ' for i in string)

#Merging json files
path = 'C:/Users/kudva/Desktop/UTA SEM 2/Final DS project/Tweets AA json'
for files in os.listdir(path):
    jsonFiles.append(files)
    #print jsonFiles
os.chdir(path)

for lines in jsonFiles:
    infile = open(lines).read()
    content = json.loads(infile)
    
    for i in range(len(content)):
        aa_tweet = remove_non_ascii(content[i]['text']).encode('utf-8')
        aa += aa_tweet + '\n'
        senseaa = TextBlob(aa_tweet)
        pola.append(senseaa.sentiment.polarity)
        subj.append(senseaa.sentiment.subjectivity)

#Code for plotting histogram with Polarity scores
plot.hist(pola, bins = 30)
plot.xlabel('Polarity Score')
plot.ylabel('Tweet Counts')
plot.grid(True)
plot.savefig('Polarity.pdf')
plot.show()

#Code for plotting histogram with Subjectivity Scores
plot.hist(subj, bins = 30)
plot.xlabel('Subjectivity Score')
plot.ylabel('Tweet Counts')
plot.grid(True)
plot.savefig('Subjectivity.pdf')
plot.show()

print('Average of Polarity Scores: {}'.format(np.mean(pola)))
print('Average of Subjectivity Scores: {}'.format(np.mean(subj)))      



with open('AllAmAirtweets.json' , 'w') as f1:
    f1.write(aa)
    


# In[2]:

# Wordcloud for 15k tweets
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
import numpy as np
from os import path
import string 

# appending words to stopwords as per judgement 
stopwords = nltk.corpus.stopwords.words('english')

stopwords.append('americanair')
stopwords.append('one')
stopwords.append('know')
stopwords.append('need')
stopwords.append('amp')
stopwords.append('indic')
stopwords.append('akhdr')
stopwords.append('lzhyaox')
stopwords.append('go')
stopwords.append('think')
stopwords.append('rt')
stopwords.append('https')
stopwords.append('co')
stopwords.append('say')
stopwords.append('us')
stopwords.append('aa')
stopwords.append('alway')
stopwords.append('hey')
stopwords.append('also')
stopwords.append('let')
stopwords.append('fli')
stopwords.append('get')
stopwords.append('like')
stopwords.append('make')
stopwords.append('rhettmc')
stopwords.append('ananavarro')
stopwords.append('airport')





#Removing punctuation and digits from AA tweets

p = string.punctuation
d = string.digits

table_p = string.maketrans(p, len(p) * " ")
table_d = string.maketrans(d, len(d) * " ")
p1= aa.translate(table_p)
p2=p1.translate(table_d)

newlist=p2.split()
#print newlist



# In[3]:

words2 = []
for w in newlist:
    if w.lower() not in stopwords and len(w) > 1:
           words2.append(w)


#Stemming process
ss = SnowballStemmer("english")
ste=[]
for words1 in words2:
    q=ss.stem(words1)
    ste.append(q)

#print ste



# In[4]:

# Wordcloud for 15k tweets
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from PIL import Image
import numpy as np
from os import path

# Read the stemmed words text
d = 'C:/Users/kudva/Desktop/UTA SEM 2/Final DS project/Tweets AA json'
text2 = ''
for word3 in ste:
    if len(word3)== 1 or word3 in stopwords:
        continue
    text2 += ' {}'.format(word3)
    
print text2

with open('airlinetest.txt', 'w') as f:
    f.write(text2)


# In[6]:

#Getting list of positive words from tweets
pos_list = set()

with open('C:/Users/kudva/Desktop/UTA SEM 2/Final DS project/Tweets AA json/positive-words.txt') as f:
    for line in f.readlines():
        if line.startswith(';'):
            continue
        pos_list.add(line.rstrip())
        
pos_list = list(pos_list)

#print type(pos_list)
pos_corpus = []

for word in ste:
      if word in pos_list:
            pos_corpus.append(word)
            
        

for word in pos_corpus:
    strwords = ','.join(pos_corpus)





# In[12]:

#Getting list of negative words from tweets

neg_list = set()

with open('C:/Users/kudva/Desktop/UTA SEM 2/Final DS project/Tweets AA json/negative-words.txt') as f:
    for line in f.readlines():
        if line.startswith(';'):
            continue
        neg_list.add(line.rstrip())
        
neg_list = list(neg_list)

neg_corpus = []

for word in ste:
      if word in neg_list:
            neg_corpus.append(word)
            

for word in neg_corpus:
    negwords = ','.join(neg_corpus)
            


    


# In[13]:

#Mask image in form of dislike button
image_mask = np.array(Image.open(path.join(d, "dislikebutton_neg.jpg")))

# Generate a word cloud image
wordcloud = WordCloud(background_color="white",max_font_size=80, max_words= 5000, mask =image_mask).generate(negwords) 

#Storing
wordcloud.to_file(path.join(d, "WCnegwordsAA.jpg"))



# Display the generated image
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.figure()
#plt.imshow(wordcloud)
#plt.imshow(image_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[21]:

#Normal wordcloud

#print negwords

negwords = set()
newneg = list(negwords)
print newneg
        


wc = WordCloud(max_font_size=45).generate(negwords) 


# Display the generated image
plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[ ]:

#Mask image in form of like button
image_mask = np.array(Image.open(path.join(d, "like_pos.jpg")))

# Generate a word cloud image
wordcloud = WordCloud(background_color="white",max_font_size=80, max_words= 5000, mask =image_mask).generate(strwords) 

#Storing
wordcloud.to_file(path.join(d, "WCposwordsAA.jpg"))



# Display the generated image
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.figure()
#plt.imshow(wordcloud)
#plt.imshow(image_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




# In[ ]:



