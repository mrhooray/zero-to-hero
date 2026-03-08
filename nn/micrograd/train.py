from micrograd.value import Value


def train(model, X, y, loss_fn, epochs=100, learning_rate=0.01, verbose=False):
    losses = []

    for epoch in range(epochs):
        total_loss = Value(0)

        for xi, yi in zip(X, y):
            inputs = [Value(x) if not isinstance(x, Value) else x for x in xi]
            target = Value(yi) if not isinstance(yi, Value) else yi
            pred = model(inputs)
            loss = loss_fn(pred, target)
            total_loss = total_loss + loss

        losses.append(total_loss.data)

        for p in model.parameters():
            p.grad = 0.0

        total_loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: loss = {total_loss.data:.6f}")

    return losses
