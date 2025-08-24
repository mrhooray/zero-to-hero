import math


class Value:
    def __init__(self, data, label="", _children=(), _op=None):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._children = _children
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, label='{self.label}', grad={self.grad})"

    def backward(self):
        visited = set()
        seq = []

        def iter(n):
            if n not in visited:
                visited.add(n)
                for c in n._children:
                    iter(c)
                seq.append(n)

        iter(self)

        self.grad = 1.0
        for n in reversed(seq):
            n._backward()

    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, "", (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return Value(other) + self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, "", (self, other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return Value(other) * self

    def __pow__(self, other):
        out = Value(self.data**other, "", (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, "", (self, other), "/")

        def _backward():
            self.grad += out.grad / other.data
            other.grad += out.grad * (-self.data / (other.data**2))

        out._backward = _backward

        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    # Activation functions
    def tanh(self):
        out = Value(math.tanh(self.data), "", (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), "", (self,), "relu")

        def _backward():
            self.grad += out.grad * (1 if self.data > 0 else 0)

        out._backward = _backward

        return out

    def sigmoid(self):
        out = Value(1 / (1 + math.exp(-self.data)), "", (self,), "sigmoid")

        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)

        out._backward = _backward

        return out

    # Mathematical functions
    def exp(self):
        out = Value(math.exp(self.data), "", (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data), "", (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward

        return out

    def abs(self):
        out = Value(abs(self.data), "", (self,), "abs")

        def _backward():
            self.grad += out.grad * (1 if self.data >= 0 else -1)

        out._backward = _backward

        return out
