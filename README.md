## LDA文本聚类笔记


While LDA and NMF have differing mathematical underpinning, 

**both algorithm are able to return the documents that belong to a topic in a corpus and the words that belong to a topic**

 LDA is based on probabilistic graphical modeling while NMF relies on linear algebra. 
 
 **Both algorithms take as input a bag of words matrix** 
 
 (i.e., each document represented as a row, with each columns containing the count of words in the corpus).
 
LDA和NMF都能够返回语料库中属于某一个topic的文档，属于某一个topic的关键字.

>NMF和LDA都需要人为指定Topic的数量


最终得到topic-word相关的矩阵与document-topic相关的矩阵

| |word 1|word 2|word n|
|topic1|0.5|0|1|
|topic2|0|0.5|0|
|-|-|-|-|

| |topoc 1|topic 2|
|document1|1|0|
|document2|0|1|
|document3|0|1|
|-|-|-|-|

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
```


---



### reference
[Topic Modeling with Scikit Learn – ML Review – Medium](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)


