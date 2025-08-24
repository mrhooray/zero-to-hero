from graphviz import Digraph


def to_dot(node):
    graph = Digraph(graph_attr={"rankdir": "LR"})

    def on_node(node):
        node_id = str(id(node))
        graph.node(
            name=node_id,
            label=f"{node.label} | data={node.data:.4f} | grad={node.grad:.4f}",
            shape="record",
        )
        if node._op:
            node_op_id = f"{node_id}_{node._op}"
            graph.node(name=node_op_id, label=node._op)
            graph.edge(node_op_id, node_id)

    def _iter(node, visited):
        if node not in visited:
            visited.add(node)
            on_node(node)
            for child in node._children:
                node_id = str(id(node))
                node_op_id = f"{node_id}_{node._op}"
                child_id = str(id(child))
                graph.edge(child_id, node_op_id)
                _iter(child, visited)

    _iter(node, set())
    return f"\n{graph.pipe(format='dot', encoding='utf-8')}"
