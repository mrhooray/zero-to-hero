import math
from ..value import Value


class TestValue:
    # Initialization tests
    def test_init(self):
        x = Value(5.0, "x")
        assert x.data == 5.0
        assert x.label == "x"

    # Arithmetic operation tests
    def test_addition(self):
        a = Value(2.0, "a")
        b = Value(3.0, "b")
        c = a + b
        assert c.data == 5.0
        assert c._op == "+"
        assert c._children == (a, b)

    def test_multiplication(self):
        a = Value(2.0, "a")
        b = Value(3.0, "b")
        c = a * b
        assert c.data == 6.0
        assert c._op == "*"
        assert c._children == (a, b)

    def test_division(self):
        a = Value(6.0, "a")
        b = Value(3.0, "b")
        c = a / b
        assert c.data == 2.0
        assert c._op == "/"
        assert c._children == (a, b)

    def test_rdiv(self):
        a = Value(2.0, label="a")
        b = 6 / a
        assert b.data == 3.0
        assert b._op == "/"
        assert len(b._children) == 2
        assert b._children[0].data == 6
        assert b._children[1] == a

    def test_power(self):
        a = Value(2.0, label="a")
        b = a**3
        assert b.data == 8.0
        assert b._op == "**3"
        assert b._children == (a,)

    def test_subtraction(self):
        a = Value(5.0, label="a")
        b = Value(3.0, label="b")
        c = a - b
        assert c.data == 2.0
        assert c._op == "+"
        assert len(c._children) == 2
        assert c._children[0] == a
        assert c._children[1].data == -3.0

    def test_negation(self):
        a = Value(2.0, label="a")
        b = -a
        assert b.data == -2.0
        assert b._op == "*"
        assert len(b._children) == 2
        assert b._children[0] == a
        assert b._children[1].data == -1

    # Reverse operation tests
    def test_radd(self):
        a = Value(2.0, label="a")
        b = 3 + a
        assert b.data == 5.0
        assert b._op == "+"
        assert len(b._children) == 2
        assert b._children[0].data == 3
        assert b._children[1] == a

    def test_rmul(self):
        a = Value(2.0, label="a")
        b = 3 * a
        assert b.data == 6.0
        assert b._op == "*"
        assert len(b._children) == 2
        assert b._children[0].data == 3
        assert b._children[1] == a

    def test_rsub(self):
        a = Value(2.0, label="a")
        b = 5 - a
        assert b.data == 3.0
        assert b._op == "+"
        assert len(b._children) == 2
        assert b._children[0].data == 5
        assert b._children[1].data == -2.0

    # Chained operations test
    def test_chained_operations(self):
        a = Value(2.0, label="a")
        b = Value(-5.0, label="b")
        c = Value(8.0, label="c")
        e = a * b
        e.label = "e"
        d = e + c
        d.label = "d"
        f = Value(-3.0, label="f")
        g = d * f
        g.label = "g"
        assert g.data == 6

    # Activation function tests
    def test_tanh(self):
        a = Value(1.0, label="a")
        b = a.tanh()
        assert abs(b.data - 0.7615941559557649) < 1e-10
        assert b._op == "tanh"
        assert b._children == (a,)

    def test_relu(self):
        a = Value(2.0, label="a")
        b = a.relu()
        assert b.data == 2.0
        assert b._op == "relu"
        assert b._children == (a,)

        c = Value(-2.0, label="c")
        d = c.relu()
        assert d.data == 0.0
        assert d._op == "relu"
        assert d._children == (c,)

    def test_sigmoid(self):
        a = Value(0.0, label="a")
        b = a.sigmoid()
        assert abs(b.data - 0.5) < 1e-10
        assert b._op == "sigmoid"
        assert b._children == (a,)

    # Mathematical function tests
    def test_exp(self):
        a = Value(1.0, label="a")
        b = a.exp()
        assert abs(b.data - math.exp(1.0)) < 1e-10
        assert b._op == "exp"
        assert b._children == (a,)

    def test_log(self):
        a = Value(2.0, label="a")
        b = a.log()
        assert abs(b.data - math.log(2.0)) < 1e-10
        assert b._op == "log"
        assert b._children == (a,)

    def test_abs(self):
        a = Value(-5.0, label="a")
        b = a.abs()
        assert b.data == 5.0
        assert b._op == "abs"
        assert b._children == (a,)

        c = Value(3.0, label="c")
        d = c.abs()
        assert d.data == 3.0
        assert d._op == "abs"
        assert d._children == (c,)
