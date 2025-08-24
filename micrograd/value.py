class Value:
    def __init__(self, data, label="", _children=(), _op=None):
        self.data = data
        self.label = label
        self._children = _children
        self._op = _op

    def __repr__(self):
        return f"{self.label}: {self.data}"

    def __add__(self, other):
        return Value(self.data + other.data, "", (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, "", (self, other), "*")
