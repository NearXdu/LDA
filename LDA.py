from preProcessing import read_csv_data
from preProcessing import generate_rows
from preProcessing import read_stopwords
from preProcessing import word_cut
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Print import display_topics




df=read_csv_data('Data/','data.csv')
#df=df.head(1000)

stopwords_list=read_stopwords(['Data/stopwords_chinese.txt','Data/stopwords_english.txt'])
rows=generate_rows(df)
corpus=word_cut(stopwords_list,rows)

no_features=2000

tf_vectorizer=CountVectorizer(max_df=0.95,min_df=2,max_features=no_features)
tf=tf_vectorizer.fit_transform(corpus)
tf_feature_names=tf_vectorizer.get_feature_names()

no_topics=200

print ('begin train....')

lda_model=LatentDirichletAllocation(n_topics=no_topics,
                              max_iter=5,
                              learning_method='online',
                              learning_offset=50.,
                              random_state=0).fit(tf)

lda_W=lda_model.transform(tf)
lda_H=lda_model.components_

no_top_words=5
no_top_documents=5

print('resuting...')


display_topics(lda_H,lda_W,tf_feature_names,rows,no_top_words,no_top_documents)
