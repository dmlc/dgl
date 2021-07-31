Latent Dirichlet Allocation
===
LDA is a classical algorithm for probabilistic graphical models. It assumes 
hierarchical Bayes models with discrete variables on sparse doc/word graphs.
This example shows how it can be done on DGL,
where the corpus is represented as a bipartite multi-graph G.
There is no back-propagation, because gradient descent is typically considered
inefficient on probability simplex.
On the provided small-scale example on 20 news groups dataset, our DGL-LDA model runs
50% faster on GPU than sklearn model without joblib parallel.

Key equations
---

The non-Bayesian form assumes document d -> topic z -> word w. The Bayesian form adds
priors and variational posteriors to the learnable parameters. In both forms, q(z|d,w)
is used as an auxiliary distribution to aid learning.

| Multinomial | p(z\|d)   | p(w\|z)  | q(z\|d,w) |
|-------------|-----------|----------|-----------|
| Parameter   | θ_d       | β_z      | ϕ_dw      |
| Prior       | Dir(α)    | Dir(η)   |           |
| Posterior   | Dir(γ_d)  | Dir(λ_z) |           |

**Non-Bayesian EM**

A simple form of topic modeling is to assume p(w|d) to be a latent-variable model, such that when the latent topic variable is unobserved, the marginal word probabilities become correlated inside the same document. Here is the marginal likelihood:

<img src="https://latex.codecogs.com/gif.latex?p(w|d)=\sum_z\theta_{dz}\beta_{zw}\quad\Rightarrow\quad&space;\log&space;p(G)=\sum_{\mbox{edge}(d,w)}\log\left(\sum_z(\theta_{dz}\beta_{zw})\right)." title="marginal" />

If we try to take a gradient against θ_d(z), we immediately realize the challenge that the denominator contains θ_d(z) in a summation term, which is nonconvex and computationally inefficient. Instead, we apply Jensen's inequality:

<img src="https://latex.codecogs.com/gif.latex?\log\left(\sum_z(\theta_{dz}\beta_{zw})\right)\geq\sum_z\phi_{dw}(z)\log\left(\frac{\theta_{dz}\beta_{zw}}{\phi_{dw}(z)}\right)=:\mbox{ELBO}(d,w)." title="elbo" />

To show more details, we write out the proof:

<img src="https://latex.codecogs.com/gif.latex?\mbox{left}-\mbox{right}=\left(\sum_z\phi_{dw}(z)\right&space;)\log\left(\sum_z(\theta_{dz}\beta_{zw})\right)-\sum_z\phi_{dw}(z)\log\left(\frac{\theta_{dz}\beta_{zw}}{\phi_{dw}(z)}\right)&space;\\&space;=\sum_z\phi_{dw}(z)\log\left(\frac{\sum_{z'}(\theta_{dz'}\beta_{z'w})}{(\theta_{dz}\beta_{zw})/\phi_{dw}(z)}&space;\right&space;)&space;=\sum_z\phi_{dw}(z)\log\left(\frac{\phi_{dw}(z)}{(\theta_{dz}\beta_{zw})/\sum_{z'}(\theta_{dz'}\beta_{z'w})}&space;\right&space;)," title="proof" />

where the last equation is nonnegative due to Kullback-Leibler divergence: KL(ϕ_dw\|\|p(z|d,w))>=0. The bound is tight when ϕ_dw is exactly the posterior distribution given the learned parameters. We thus alternate between (θ_d, β_z) and ϕ_dw. Specifically, ϕ_dw may be found via edge update and, with the ELBO, the node parameters assume cross-entropy likelihoods:

<img src="https://latex.codecogs.com/gif.latex?\mbox{ELBO}=\sum_{d,z}\left(\sum_w\phi_{dw}(z)\right)\log(\theta_{dz})+\sum_{z,w}\left(\sum_d\phi_{dw}(z)\right)\log(\beta_{zw})-\mbox{Const.}" title="node" />

To update the nodes, we simplify normalize the fractional membership ϕ_dw to find the parameters of the generative multinomial distributions.

**Variational Bayes**

A Bayesian model adds Dirichlet priors to θ_d & β_z. This causes the posterior to be implicit and the bound to be loose. We will still use an independence assumption and cycle through the variational parameters similarly to coordinate ascent.

 * The evidence lower-bound is

 <img src="https://latex.codecogs.com/gif.latex?\log&space;p(G)=\sum_{(d,w)}\log\left(\int_{\theta_d}\sum_z\int_{\beta_z}(\theta_{dz}\beta_{zw}){\rm\,d}P(\beta_z;\eta){\rm\,d}P(\theta_d;\alpha)\right)&space;\\&space;\geq&space;\mathbb{E}_q\left[\sum_{(d,w)}\log\left(&space;\frac{\theta_{dz}\beta_{zw}}&space;{q(z;\phi_{dw})}&space;\right)&space;&plus;\sum_{d}&space;\log\left(&space;\frac{p(\theta_d;\alpha)}{q(\theta_d;\gamma_d)}&space;\right)&space;&plus;\sum_{z}&space;\log\left(&space;\frac{p(\beta_z;\eta)}{q(\beta_z;\lambda_z)}&space;\right)\right]" title="elbo2" />

 * ELBO objective function factors as

 <img src="https://latex.codecogs.com/gif.latex?\sum_{(d,w)}&space;\phi_{dw}^{\top}\left(&space;\mathbb{E}_{\gamma_d}[\log\theta_d]&space;&plus;\mathbb{E}_{\lambda}[\log\beta_{:w}]&space;-\log\phi_{dw}&space;\right)&space;\\&space;&plus;&space;\sum_d&space;(\alpha-\gamma_d)^\top\mathbb{E}_{\gamma_d}[\log&space;\theta_d]-(\log&space;B(\alpha)-\log&space;B(\gamma_d))&space;\\&space;&plus;&space;\sum_z&space;(\eta-\lambda_z)^\top\mathbb{E}_{\lambda_z}[\log&space;\beta_z]-(\log&space;B(\eta)-\log&space;B(\lambda_z))" title="factors" />

 * Similarly, optimization alternates between ϕ, γ, λ. Since θ, β are random, we use an explicit solution for E[log X] under Dirichlet distribution via digamma function.

DGL usage
---
The corpus is represented as a bipartite multi-graph G.
We use DGL to propagate information through the edges and aggregate the distributions at doc/word nodes.
For scalability, the phi variables are transient and updated during message passing.
The gamma / lambda variables are updated after the nodes receive all edge messages.
Following the conventions in [1], the gamma update is called E-step and the lambda update is called M-step.
The lambda variable is further recorded by the trainer.
A separate function is used to produce perplexity, which is based on the ELBO objective function divided by the total numbers of word/doc occurrences.

Example
---
`%run example_20newsgroups.py`

 * Approximately matches scikit-learn training perplexity after 10 rounds of training.
 * Exactly matches scikit-learn training perplexity if word_z is set to lda.components_.T
 * There is a difference in how we compute testing perplexity. We weigh the beta contributions by the training word counts, whereas sklearn weighs them by test word counts.
 * The DGL-LDA model runs 50% faster on GPU devices compared with sklearn without joblib parallel.

Advanced configurations
---
 * Set `0<lr<1` for online learning with partial_fit.
 * Set `lr["doc"]>=100` or `lr["word"]>=100` to disable the corresponding Bayesian priors.

References
---

1. Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
Dirichlet Allocation. Advances in Neural Information Processing Systems 23
(NIPS 2010).
2. Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
