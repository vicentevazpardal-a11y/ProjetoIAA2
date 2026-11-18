import numpy as np

class ArvoreDecisao:
    class Node:
        def __init__(self, feature=None, threshold=None, esquerda=None, direita=None, *, valor=None):
            self.feature = feature
            self.limiar = threshold
            self.esquerda = esquerda
            self.direita = direita
            self.valor = valor
            self.value = valor

        def is_leaf_node(self):
            return self.value is not None

    def __init__(self, profundidade_maxima=None):
        self.profundidade_maxima = profundidade_maxima
        self.raiz = None

    def _entropy(self, y):
        classes, contagens = np.unique(y, return_counts=True)
        probabilidades = contagens / len(y)
        entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-9))
        return entropia

    def _mutual_information(self, y, y_esquerda, y_direita):
        H_parent = self._entropy(y)
        n = len(y)
        n_esq, n_dir = len(y_esquerda), len(y_direita)
        
        if n_esq == 0 or n_dir == 0:
            return 0.0
            
        e_esq = self._entropy(y_esquerda)
        e_dir = self._entropy(y_direita)
        
        child_entropy = (n_esq / n) * e_esq + (n_dir / n) * e_dir
        return H_parent - child_entropy

    def _best_split(self, X, y):
        n_amostras, n_features = X.shape
        if n_amostras <= 1:
            return None, None

        melhor_mi = -1
        idx_melhor_feature = None
        melhor_limiar = None

        for feat_idx in range(n_features):
            X_coluna = X[:, feat_idx]
            limiares = np.unique(X_coluna)

            for limiar in limiares:
                mask_esq = X_coluna <= limiar
                y_esq, y_dir = y[mask_esq], y[~mask_esq]

                if len(y_esq) == 0 or len(y_dir) == 0:
                    continue

                mi = self._mutual_information(y, y_esq, y_dir)

                if mi > melhor_mi:
                    melhor_mi = mi
                    idx_melhor_feature = feat_idx
                    melhor_limiar = limiar

        return idx_melhor_feature, melhor_limiar

    def _classe_maioritaria(self, y):
        classes, contagens = np.unique(y, return_counts=True)
        max_contagem = np.max(contagens)
        # Em caso de empate, escolhe a classe correspondente ao maior valor
        return np.max(classes[contagens == max_contagem])

    def _build_tree(self, X, y, profundidade=0):
        n_amostras, n_features = X.shape
        n_classes = len(np.unique(y))

        if (self.profundidade_maxima is not None and profundidade >= self.profundidade_maxima) or \
           n_classes == 1 or n_amostras == 0:
            return self.Node(valor=self._classe_maioritaria(y))

        feat_idx, limiar = self._best_split(X, y)

        if feat_idx is None or self._mutual_information(y, y[X[:, feat_idx] <= limiar], y[X[:, feat_idx] > limiar]) <= 0:
            return self.Node(valor=self._classe_maioritaria(y))

        mask_esq = X[:, feat_idx] <= limiar
        esquerda = self._build_tree(X[mask_esq], y[mask_esq], profundidade + 1)
        direita = self._build_tree(X[~mask_esq], y[~mask_esq], profundidade + 1)

        return self.Node(feature=feat_idx, threshold=limiar, esquerda=esquerda, direita=direita)

    def fit(self, X, y):
        self.raiz = self._build_tree(X, y)

    def _predict_row(self, node, row):
        if node.is_leaf_node():
            return node.value
        if row[node.feature] <= node.limiar:
            return self._predict_row(node.esquerda, row)
        return self._predict_row(node.direita, row)

    def predict(self, X):
        return np.array([self._predict_row(self.raiz, row) for row in X])