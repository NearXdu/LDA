import pandas as pd
from itertools import chain
import jieba

def hello():
    print('hello world')

def read_csv_data(filePath='~/Data/',fileName='default.csv'):
    csv=filePath+fileName
    df=pd.read_csv(csv,sep='delimiter',engine='python')
    return df

def read_stopwords(stopwords_file_list):
    stopword_lists=[]
    # read
    for item in stopwords_file_list:
        stopword_list={}.fromkeys([line.strip() for line in open(item)])
        stopword_lists.append(stopword_list)
    # merge
    stopword_lists=list(chain(*stopword_lists))
    # remove dups
    stopword_lists=list(set(stopword_lists))
    return stopword_lists

def generate_rows(df):
    rows=[]
    for row in df.itertuples(index=True,name='Pandas'):
        rowStr=row[1]
        rowList=rowStr.split(',')
        newRow=''
        for index in range(1,len(rowList)-1):
            newRow+=rowList[index]

        rows.append(newRow)
    return rows

def word_cut(stopword_list,rows):
    corpus=[]
    for row in rows:
        segs=jieba.cut(row,cut_all=False)
        segs=[word.encode('utf-8') for word in list(segs)]
        segs=[word for word in list(segs) if word not in stopword_list]
        temp=" ".join(segs)
        corpus.append(temp)
    return corpus

