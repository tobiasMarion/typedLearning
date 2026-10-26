# Algorithms

Each algorithm has its own document with the math derivation, implementation decisions, and what was learned along the way.

| #   | Algorithm                                 | Topics                                                          | Status            |
| --- | ----------------------------------------- | --------------------------------------------------------------- | ----------------- |
| 1   | [Linear regression](linear-regression.md) | MSE, gradients, gradient descent, scaling, closed-form solution | MVP               |
| 2   | Logistic regression                       | Sigmoid, log-loss, threshold, L1/L2 regularization              | MVP (or KNN)      |
| 3   | KNN                                       | Distances, voting, averaging, curse of dimensionality           | MVP (or logistic) |
| 4   | Decision trees                            | Gini/entropy, splits, stopping criteria, pruning                | later             |
| 5   | Neural networks                           | Forward pass, chain rule, backprop, minibatch, momentum, Adam   | later             |
| 6   | Ensembles                                 | Random forest, gradient boosting                                | later             |

The order is a proposal, not a requirement.

## Template

Algorithm documents follow this structure:
1. **Intuition:** what the algorithm does, in one sentence.
2. **Model:** the prediction function.
3. **Loss:** the loss function and its derivative.
4. **Training:** the update rule and the stopping criterion.
5. **Hyperparameters:** the list and the effect of each one.
6. **Pitfalls:** scaling, convergence, degenerate cases.
7. **Oracle:** what the implementation is tested against.
8. **Learning notes.**
