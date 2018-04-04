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
Assume there are N words in the corpus, for each word $v_i$ has appeard $n_i$ times, $\vec{n}=(n_1,n_2,...,n_v)$ will be a Multinomial Distribution
$$p(\vec{n})=Mult(\vec{n}|\vec{p},N)=\binom{N}{\vec{n}}\prod_{k=1}^Vp_k^{n_k}$$
Now the propability of corpus is
$$p(W)=p(\vec{w_1}\vec{w_2}...\vec{w_m})=\prod_{k=1}^Vp_k^{n_k}$$
So $$\hat{p_i}=\frac{n_i}{N}$$

## Bayes Unigram Model
$$\hat{p_i}=\frac{n_i+\alpha_i}{\Sigma_{i=1}^V(n_i+\alpha_i)}$$

## LDA Model 
### Two most important  formula
$$\vec{\alpha}\to\vec{\theta}_m\to z_{m,n}$$
$$\vec{\beta}\to\vec{\varphi}_k\to w_{m,n}|k=z_{m,n}$$

### Gibbs Sampling 
$$p(z_i=k|\vec{z}_{\lnot i},\vec{w}) \varpropto \frac{n_{m,\lnot i}^{(k)}+\alpha_k}{\Sigma_{k=1}^K(n_{m,\lnot i}^{(t)}+\alpha_k)}\cdot\frac{n_{k,\lnot i}^{(t)}+\beta_t}{\Sigma_{t=1}^V(n_{k,\lnot i}^{(t)}+\beta_t)}$$

## TWLDA
