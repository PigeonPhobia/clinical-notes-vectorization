import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.manifold import TSNE

from src import preprocess


def plot_category_pie_chart(df):
    df['category'].value_counts().plot.pie(ylabel='')

def plot_char_length_hist(notes):
    len_notes = [len(note) for note in notes]
    plt.hist(len_notes, bins=30)
    plt.title('Number of characters')
    plt.show()
    
    
def plot_token_length_hist(notes_doc):
    len_notes = [len(note) for note in notes_doc]
    plt.hist(len_notes, bins=30)
    plt.title('Number of tokens')
    plt.show()
    
    
def plot_wordcloud(notes, freq_measure='count'):
    wc = WordCloud(width=1280, height=720, background_color="white", max_words=200, contour_width=0, random_state=46)
    
    if freq_measure == 'count':
        matrix, vocab = preprocess.count_vectorize(preprocess.join_as_sentence(notes))
        word_freqs = dict(zip(vocab, matrix.A.sum(axis=0)))
    elif freq_measure == 'tfidf':
        matrix, vocab = preprocess.count_vectorize(preprocess.join_as_sentence(notes))
        word_freqs = dict(zip(vocab, matrix.A.max(axis=0)))
    else:
        return
    
    wc.generate_from_frequencies(word_freqs)

    plt.figure(dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    
def plot_tsne(vectors, labels):    
    tsne_model_2d = TSNE(n_components=2, n_iter=1000, perplexity=10)
    vectors_2d = tsne_model_2d.fit_transform(vectors)
    
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(vectors_2d[idx][:,0], vectors_2d[idx][:,1], label=str(label))
    ax.legend(loc='upper right')
    plt.axis('off')
    plt.show()
    

def plot_cluster_wordcloud(centroids, vocab, shape):
    # to get a circle shape for word cloud
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    
    wc = WordCloud(background_color="white", max_words=50, contour_width=0, random_state=46, mask=mask)

    for i in range(shape[0]*shape[1]):
        plt.subplot(shape[0], shape[1], i+1)
        
        centroid = centroids[i]
        indices = np.nonzero(centroid)[0]
        word_freqs = dict(zip(vocab[indices], centroid[indices]))

        wc.generate_from_frequencies(word_freqs)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()