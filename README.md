TWLDA (Term Weighting LDA)
===

TWLDA is a new topic of LDA which assigns low weights to words with low topic discriminating power. For more details, please refer to "Exploring Topic Discriminating Power of Words in Latent Dirichlet
Allocation" created by Kai Yang, Yi Cai and Zhenhong Chen.

## MCMC sample algorithm

<a herf="https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo">Markov chain Monte Carlo</a>

## Metropolis-Hastings algorithm

<a herf="https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm">Metropolis–Hastings algorithm</a>

## Unigram Model
$\vec{p} = (p_1, p_2, ..., p_v)$ refers to the probability of each word to be chosed.
So the result is $w \sim Mult(w|\vec{p})$
For a doc created using Unigram Model, the probability that the doc to be created is
$$p(\vec{w}) = p(w_1, w_2, ..., w_n) = p(w_1)p(w_2)...p(w_n)$$
And for the hypothesis that each doc is independent, a corpus with some doc $W = (\vec{w_1}),(\vec{w_2}),...,(\vec{w_m})$, the probability is
$$p(W) = p(\vec{w_1})p(\vec{w_2})...p(\vec{w_m})$$
