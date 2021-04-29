Latent Dirichlet Allocation
===
LDA is a classical algorithm for probabilistic graphical models. It assumes 
hierarchical Bayes models with discrete variables on sparse doc/word graphs.
This example shows how it can be done on DGL.
There is no back-propagation, because gradient descent is typically considered
inefficient on probability simplex.

Key equations
---

 * A corpus is represented as a multi-graph: document(d) -> topic(z) -> word(w)
 * Document -> topic distribution
 <img src="https://latex.codecogs.com/gif.latex?\theta_d&space;\sim&space;Dir(\alpha)" title="theta_d" />
 * Topic -> word distribution
 <img src="https://latex.codecogs.com/gif.latex?\beta_z&space;\sim&space;Dir(\eta)" title="\beta_z" />

**MAP**
A non-Bayesian solution is just an inner product to integrate out some latent variable:
<img src="https://latex.codecogs.com/gif.latex?p(G(d\stackrel{z}{\to}w))=\prod_{(d,w)}\left(&space;\sum_z\theta_{dz}\beta_{zw}\right)." title="map" />
The complication is in the variable sharing in different doc/word combinations and the fact that \theta_d and \beta_z need to stay in probability simplex.

**Variational Bayes**

 * Define variational q-distributions to be independent
 <img src="https://latex.codecogs.com/gif.latex?q(z_{dw};\phi_{dw}),&space;q(\theta_d;\gamma_d),&space;q(\beta_z;\lambda_z)" title="q" />

 * The evidence lower-bound is
 <img src="https://latex.codecogs.com/gif.latex?\log&space;p(G(d\stackrel{z}{\to}w))\geq&space;\mathbb{E}_q\left[\sum_{(d,w)}\log\left(&space;\frac{\theta_{dz}\beta_{zw}}{q(z)}&space;\right)&space;&plus;\sum_{d}&space;\log\left(&space;\frac{p(\theta_d)}{q(\theta_d)}&space;\right)&space;&plus;\sum_{z}&space;\log\left(&space;\frac{p(\beta_z)}{q(\beta_z)}&space;\right)\right]" title="elbo" />

 * ELBO factors as
 <img src="https://latex.codecogs.com/gif.latex?\sum_{(d,w)}&space;\phi_{dw}^{\top}\left(&space;\mathbb{E}_q[\log\theta_d]&space;&plus;\mathbb{E}_q[\log\beta_{:w}]&space;-\log\phi_{dw}&space;\right)&space;&plus;&space;\sum_d&space;(\alpha-\gamma_d)^\top\mathbb{E}_q[\log&space;\theta_d]-\log(B(\alpha)-B(\gamma_d))&space;&plus;&space;\sum_z&space;(\eta-\lambda_z)^\top\mathbb{E}_q[\log&space;\beta_z]-\log(B(\eta)-B(\lambda_z))" title="factors" />

DGL usage
---
We use DGL to propagate the information through edges to aggregate the distributions in doc/word nodes.
The phi variables are updated during message passing.
The theta / beta variables are updated after the nodes receive all edge messages.
A separate function is used to produce perplexity per occurrence of word/doc tuple.

Example
---
`%run example_20newsgroups.py`
 * Approximately matches scikit-learn training perplexity after 10 rounds of training.
 * Exactly matches scikit-learn training perplexity if word_z is set to lda.components_
 * I think there is a bug in scikit-learn that testing perplexity(beta) is not correctly normalized.

Advanced configurations
---
 * Set `step_size['word']=100` to obtain a MAP result on beta.
 * Set `0<word_rho<1` for online learning
