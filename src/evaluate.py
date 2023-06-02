from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from src import preprocess

def print_topk_most_similar_notes(df, vectors, idx, k):
    most_similar = cosine_similarity(vectors[[idx]], vectors).squeeze().argsort()[::-1][:k+1]
    print('QUERY CLINICAL NOTE')
    print('Category:', df.iloc[idx]['category'])
    print()
    preprocess.print_note(df.iloc[idx]['notes'])
    print()
    
    for i in most_similar[1:]:
        print('******************************************************************************************')
        print('Category:', df.iloc[i]['category'])
        print()
        preprocess.print_note(df.iloc[i]['notes'])
        print()
        
        
def kmeans_cluster(k, vectors):
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(vectors)