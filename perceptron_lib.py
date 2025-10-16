import math
import random
from typing import List, Tuple, Union

# ---- вспомогательные функции ----

def zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows: int, cols: int, scale: float = 1.0) -> List[List[float]]:
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]


def dot(a: List[float], b: List[List[float]]) -> List[float]:
    m = len(b[0])
    res = [0.0 for _ in range(m)]
    for i, ai in enumerate(a):
        for j in range(m):
            res[j] += ai * b[i][j]
    return res


def add_bias(vec: List[float], bias: List[float]) -> List[float]:
    return [v + b for v, b in zip(vec, bias)]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def sigmoid_vec(v: List[float]) -> List[float]:
    return [sigmoid(x) for x in v]


def sigmoid_derivative_from_act(a: float) -> float:
    return a * (1 - a)


def softmax(v: List[float]) -> List[float]:
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_loss(probs: List[float], target_index: int) -> float:
    p = max(1e-15, probs[target_index])
    return -math.log(p)

# ---- MLP class ----

class MLP:
    def __init__(
        self,
        layers: Union[List[int], None] = None,
        input_dim: int = None,
        hidden_layers: Union[List[int], None] = None,
        output_dim: int = None,
        seed: int = None,
        l2_lambda: float = 0.0,
        dropout: float = 0.0,
    ):
        if seed is not None:
            random.seed(seed)

        # Позволяет задавать архитектуру двумя способами: напрямую или по частям
        if layers is None:
            if input_dim is None or output_dim is None:
                raise ValueError("Either 'layers' or ('input_dim' and 'output_dim') must be specified.")
            hidden_layers = hidden_layers or []
            layers = [input_dim] + hidden_layers + [output_dim]

        self.layers = layers
        self.num_layers = len(layers)
        self.l2_lambda = l2_lambda
        self.dropout = dropout

        # Инициализация весов и смещений
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            in_size = layers[i]
            out_size = layers[i + 1]
            scale = math.sqrt(2.0 / max(1, in_size))
            self.weights.append(random_matrix(in_size, out_size, scale))
            self.biases.append([0.0 for _ in range(out_size)])

    def forward(self, x: List[float], train_mode: bool = False) -> Tuple[List[List[float]], List[List[float]]]:
        activations = [x[:]]
        pre_acts = []
        for i in range(self.num_layers - 1):
            W = self.weights[i]
            b = self.biases[i]
            z = add_bias(dot(activations[-1], W), b)
            pre_acts.append(z)
            if i == self.num_layers - 2:
                a = softmax(z)
            else:
                a = sigmoid_vec(z)
                if train_mode and self.dropout > 0.0:
                    # применяем простой dropout mask и масштабирование для сохранения ожидания
                    mask = [(1.0 if random.random() > self.dropout else 0.0) for _ in range(len(a))]
                    a = [a[j] * mask[j] / (1.0 - self.dropout) for j in range(len(a))]
            activations.append(a)
        return activations, pre_acts

    def predict_proba(self, x: List[float]) -> List[float]:
        acts, _ = self.forward(x, train_mode=False)
        return acts[-1]

    def predict(self, x: List[float]) -> int:
        probs = self.predict_proba(x)
        return max(range(len(probs)), key=lambda i: probs[i])

    def _one_hot(self, k: int, size: int) -> List[float]:
        v = [0.0] * size
        v[k] = 1.0
        return v

    def train(self, X: List[List[float]], y: List[int], epochs: int = 100, lr: float = 0.1, batch_size: int = 16, verbose: bool = True):
        n = len(X)
        out_size = self.layers[-1]
        for epoch in range(1, epochs + 1):
            indices = list(range(n))
            random.shuffle(indices)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                grad_w = [zeros(len(self.weights[i]), len(self.weights[i][0])) for i in range(len(self.weights))]
                grad_b = [[0.0 for _ in range(len(self.biases[i]))] for i in range(len(self.biases))]

                for idx in batch_idx:
                    x = X[idx]
                    target = y[idx]
                    acts, pre_acts = self.forward(x, train_mode=True)
                    probs = acts[-1]
                    total_loss += cross_entropy_loss(probs, target)

                    y_onehot = self._one_hot(target, out_size)
                    delta = [probs[i] - y_onehot[i] for i in range(out_size)]

                    deltas = [None] * (self.num_layers - 1)
                    deltas[-1] = delta

                    for l in range(self.num_layers - 3, -1, -1):
                        W_next = self.weights[l + 1]
                        delta_next = deltas[l + 1]
                        size_j = self.layers[l + 1]
                        delta_l = [0.0] * size_j
                        for j in range(size_j):
                            s = 0.0
                            for k in range(len(delta_next)):
                                s += W_next[j][k] * delta_next[k]
                            act_j = acts[l + 1][j]
                            delta_l[j] = s * sigmoid_derivative_from_act(act_j)
                        deltas[l] = delta_l

                    for l in range(self.num_layers - 1):
                        a_prev = acts[l]
                        delta_l = deltas[l]
                        for i_neu, ai in enumerate(a_prev):
                            for j_neu in range(len(delta_l)):
                                grad_w[l][i_neu][j_neu] += ai * delta_l[j_neu]
                        for j_neu in range(len(delta_l)):
                            grad_b[l][j_neu] += delta_l[j_neu]

                bs = len(batch_idx)
                if bs == 0:
                    continue
                for l in range(self.num_layers - 1):
                    for i_neu in range(len(self.weights[l])):
                        for j_neu in range(len(self.weights[l][0])):
                            dw = grad_w[l][i_neu][j_neu] / bs + self.l2_lambda * self.weights[l][i_neu][j_neu]
                            self.weights[l][i_neu][j_neu] -= lr * dw
                    for j_neu in range(len(self.biases[l])):
                        self.biases[l][j_neu] -= lr * (grad_b[l][j_neu] / bs)

            avg_loss = total_loss / n
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    def accuracy(self, X: List[List[float]], y: List[int]) -> float:
        correct = 0
        for xi, yi in zip(X, y):
            if self.predict(xi) == yi:
                correct += 1
        return correct / len(X) if X else 0.0


if __name__ == "__main__":
    import math
    import random

    def generate_gaussian_clusters(n_per_class: int = 200, dim: int = 2):
        X, y = [], []
        for i in range(n_per_class):
            X.append([random.gauss(-1.0, 0.6) for _ in range(dim)])
            y.append(0)
        for i in range(n_per_class):
            X.append([random.gauss(1.0, 0.6) for _ in range(dim)])
            y.append(1)
        return X, y

    def normalize(X: List[List[float]]) -> List[List[float]]:
        dim = len(X[0])
        mean = [sum(x[j] for x in X) / len(X) for j in range(dim)]
        std = [math.sqrt(sum((x[j]-mean[j])**2 for x in X) / len(X)) for j in range(dim)]
        # avoid division by zero
        std = [s if s > 0 else 1.0 for s in std]
        return [[(x[j]-mean[j])/std[j] for j in range(dim)] for x in X]

    # main demo dataset
    X, y = generate_gaussian_clusters(250)
    X = normalize(X)
    combined = list(zip(X, y))
    random.shuffle(combined)
    split = int(len(combined)*0.8)
    train = combined[:split]
    test = combined[split:]

    # fixed bug: correctly unpack train/test into features and labels
    X_train, y_train = [a for a, _ in train], [label for _, label in train]
    X_test, y_test = [a for a, _ in test], [label for _, label in test]

    model = MLP(input_dim=2, hidden_layers=[16, 8], output_dim=2, seed=42, l2_lambda=0.001, dropout=0.1)
    print("Training MLP with architecture:", model.layers)
    model.train(X_train, y_train, epochs=100, lr=0.5, batch_size=32)

    test_acc = model.accuracy(X_test, y_test)
    print("Test accuracy:", test_acc)

    # --- Additional smoke tests (automated checks) ---
    # 1) probabilites sum to 1
    sample = X_test[0]
    probs = model.predict_proba(sample)
    assert abs(sum(probs) - 1.0) < 1e-6, "Softmax probabilities must sum to 1"

    # 2) accuracy on this simple separable dataset should be reasonably high
    assert test_acc >= 0.7, f"Expected test accuracy >= 0.7 on synthetic dataset, got {test_acc}"

    # 3) small architecture sanity: create tiny model and ensure it runs
    small_model = MLP(layers=[2, 4, 2], seed=1)
    small_model.train(X_train[:40], y_train[:40], epochs=5, lr=0.1, batch_size=8, verbose=False)
    print("All smoke tests passed.")
