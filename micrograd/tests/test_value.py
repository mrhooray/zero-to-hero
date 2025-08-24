from ..value import Value


class TestValue:
    def test_init(self):
        x = Value(5.0, "x")
        assert x.data == 5.0
        assert x.label == "x"

    def test_addition(self):
        a = Value(2.0, "a")
        b = Value(3.0, "b")
        c = a + b
        assert c.data == 5.0
        assert c._children == (a, b)
        assert c._op == "+"

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
