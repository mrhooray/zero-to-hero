import math


class Value:
    def __init__(self, data, label="", _children=(), _op=None):
        self.data = data
        self.label = label
        self._children = _children
        self._op = _op

    def __repr__(self):
        return f"{self.label}: {self.data}"

    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, "", (self, other), "+")

    def __radd__(self, other):
        return Value(other) + self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, "", (self, other), "*")

    def __rmul__(self, other):
        return Value(other) * self

    def __pow__(self, other):
        return Value(self.data**other, "", (self,), f"**{other}")

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data / other.data, "", (self, other), "/")

    def __rtruediv__(self, other):
        return Value(other) / self

    # Activation functions
    def tanh(self):
        return Value(math.tanh(self.data), "", (self,), "tanh")

    def relu(self):
        return Value(max(0, self.data), "", (self,), "relu")

    def sigmoid(self):
        return Value(1 / (1 + math.exp(-self.data)), "", (self,), "sigmoid")

    # Mathematical functions
    def exp(self):
        return Value(math.exp(self.data), "", (self,), "exp")

    def log(self):
        return Value(math.log(self.data), "", (self,), "log")

    def abs(self):
        return Value(abs(self.data), "", (self,), "abs")
