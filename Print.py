import numpy as np


def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print "Topic %d: " % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print documents[doc_index]
