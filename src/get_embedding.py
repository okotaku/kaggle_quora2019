import numpy as np


class GetEmbedding():
    def __init__(self, max_features, word_index):
        self.max_features = max_features
        self.word_index = word_index

    def load(self, embedding_file, emb_mean=None, emb_std=None):
        embeddings_index = dict(
            self._get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if
            len(o) > 100)
        all_embs = np.stack(embeddings_index.values())
        if emb_mean is None: all_embs.mean()
        if emb_std is None: all_embs.std()
        embed_size = all_embs.shape[1]

        nb_words = min(self.max_features, len(self.word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def _get_coefs(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')
