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
For larger graphs, thanks to subgraph sampling and low-memory implementation, we may fit 100 million unique words with 256 topic dimensions on a large multi-gpu machine.
(The runtime memory is often less than 2x of parameter storage.)


Key equations
---

<!-- https://editor.codecogs.com/ -->

Let k be the topic index variable with one-hot encoded vector representation z. The rest of the variables are:

|             | z_d\~p(θ_d) | w_k\~p(β_k) | z_dw\~q(ϕ_dw) |
|-------------|-------------|-------------|---------------|
| Prior       | Dir(α)      | Dir(η)      |     (n/a)     |
| Posterior   | Dir(γ_d)    | Dir(λ_k)    |     (n/a)     |

We overload w with bold-symbol-w, which represents the entire observed document-world multi-graph. The difference is better shown in the original paper.

**Multinomial PCA**

Multinomial PCA is a "latent allocation" model without the "Dirichlet".
Its data likelihood sums over the latent topic-index variable k,
<img src="https://latex.codecogs.com/svg.image?\inline&space;p(w_{di}|\theta_d,\beta)=\sum_k\theta_{dk}\beta_{kw}"/>,
where θ_d and β_k are shared within the same document and topic, respectively.

If we perform gradient descent, we may need additional steps to project the parameters to the probability simplices:
<img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_k\theta_{dk}=1"/>
and
<img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_w\beta_{kw}=1"/>.
Instead, a more efficient solution is to borrow ideas from evidence lower-bound (ELBO) decomposition:

<!--
\log p(w) \geq \mathcal{L}(w,\phi)
\stackrel{def}{=}
\mathbb{E}_q [\log p(w,z;\theta,\beta) - \log q(z;\phi)]
\\=
\mathbb{E}_q [\log p(w|z;\beta) + \log p(z;\theta) - \log q(z;\phi)]
\\=
\sum_{dwk}n_{dw}\phi_{dwk} [\log\beta_{kw} + \log \theta_{dk} - \log \phi_{dwk}]
-->

<img src="https://latex.codecogs.com/svg.image?\log&space;p(w)&space;\geq&space;\mathcal{L}(w,\phi)\stackrel{def}{=}\mathbb{E}_q&space;[\log&space;p(w,z;\theta,\beta)&space;-&space;\log&space;q(z;\phi)]\\=\mathbb{E}_q&space;[\log&space;p(w|z;\beta)&space;&plus;&space;\log&space;p(z;\theta)&space;-&space;\log&space;q(z;\phi)]\\=\sum_{dwk}n_{dw}\phi_{dwk}&space;[\log\beta_{kw}&space;&plus;&space;\log&space;\theta_{dk}&space;-&space;\log&space;\phi_{dwk}]"/>

The solutions for
<img src="https://latex.codecogs.com/svg.image?\inline&space;\theta_{dk}\propto\sum_wn_{dw}\phi_{dwk}"/>
and
<img src="https://latex.codecogs.com/svg.image?\inline&space;\beta_{kw}\propto\sum_dn_{dw}\phi_{dwk}"/>
follow from the maximization of cross-entropy loss.
The solution for
<img src="https://latex.codecogs.com/svg.image?\inline&space;\phi_{dwk}\propto&space;\theta_{dk}\beta_{kw}"/>
follows from Kullback-Leibler divergence.
After normalizing to
<img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_k\phi_{dwk}=1"/>,
the difference
<img src="https://latex.codecogs.com/svg.image?\inline&space;\ell_{dw}=\log\beta_{kw}+\log\theta_{dk}-\log\phi_{dwk}"/>
becomes constant in k,
which is connected to the likelihood for the observed document-word pairs.

Note that after learning, the document vector θ_d considers the correlation between all words in d and similarly the topic distribution vector β_k considers the correlations in all observed documents.

**Variational Bayes**

A Bayesian model adds Dirichlet priors to θ_d and β_z, which leads to a similar ELBO if we assume independence
<img src="https://latex.codecogs.com/svg.image?\inline&space;q(z,\theta,\beta;\phi,\gamma,\lambda)=q(z;\phi)q(\theta;\gamma)q(\beta;\lambda)"/>,
i.e.:

<!--
\log p(w;\alpha,\eta) \geq \mathcal{L}(w,\phi,\gamma,\lambda)
\stackrel{def}{=}
\mathbb{E}_q [\log p(w,z,\theta,\beta;\alpha,\eta) - \log q(z,\theta,\beta;\phi,\gamma,\lambda)]
\\=
\mathbb{E}_q \left[
\log p(w|z,\beta) + \log p(z|\theta) - \log q(z;\phi)
+\log p(\theta;\alpha) - \log q(\theta;\gamma)
+\log p(\beta;\eta) - \log q(\beta;\lambda)
\right]
\\=
\sum_{dwk}n_{dw}\phi_{dwk} (\mathbb{E}_{\lambda_k}[\log\beta_{kw}] + \mathbb{E}_{\gamma_d}[\log \theta_{dk}] - \log \phi_{dwk})
\\+\sum_{d}\left[
(\alpha-\gamma_d)^\top\mathbb{E}_{\gamma_d}[\log\theta_d]
-(\log B(\alpha 1_K) - \log B(\gamma_d))
\right]
\\+\sum_{k}\left[
(\eta-\lambda_k)^\top\mathbb{E}_{\lambda_k}[\log\beta_k]
-(\log B(\eta 1_W) - \log B(\lambda_k))
\right]
 -->

<img src="https://latex.codecogs.com/svg.image?\log&space;p(w;\alpha,\eta)&space;\geq&space;\mathcal{L}(w,\phi,\gamma,\lambda)\stackrel{def}{=}\mathbb{E}_q&space;[\log&space;p(w,z,\theta,\beta;\alpha,\eta)&space;-&space;\log&space;q(z,\theta,\beta;\phi,\gamma,\lambda)]\\=\mathbb{E}_q&space;\left[\log&space;p(w|z,\beta)&space;&plus;&space;\log&space;p(z|\theta)&space;-&space;\log&space;q(z;\phi)&plus;\log&space;p(\theta;\alpha)&space;-&space;\log&space;q(\theta;\gamma)&plus;\log&space;p(\beta;\eta)&space;-&space;\log&space;q(\beta;\lambda)\right]\\=\sum_{dwk}n_{dw}\phi_{dwk}&space;(\mathbb{E}_{\lambda_k}[\log\beta_{kw}]&space;&plus;&space;\mathbb{E}_{\gamma_d}[\log&space;\theta_{dk}]&space;-&space;\log&space;\phi_{dwk})\\&plus;\sum_{d}\left[(\alpha-\gamma_d)^\top\mathbb{E}_{\gamma_d}[\log\theta_d]-(\log&space;B(\alpha&space;1_K)&space;-&space;\log&space;B(\gamma_d))\right]\\&plus;\sum_{k}\left[(\eta-\lambda_k)^\top\mathbb{E}_{\lambda_k}[\log\beta_k]-(\log&space;B(\eta&space;1_W)&space;-&space;\log&space;B(\lambda_k))\right]"/>


**Solutions**

The solutions to VB subsumes the solutions to multinomial PCA when n goes to infinity.
The solution for ϕ is
<img src="https://latex.codecogs.com/svg.image?\inline&space;\log\phi_{dwk}=\mathbb{E}_{\gamma_d}[\log\theta_{dk}]+\mathbb{E}_{\lambda_k}[\log\beta_{kw}]-\ell_{dw}"/>,
where the additional expectation can be expressed via digamma functions
and
<img src="https://latex.codecogs.com/svg.image?\inline&space;\ell_{dw}=\log\sum_k\exp(\mathbb{E}_{\gamma_d}[\log\theta_{dk}]+\mathbb{E}_{\lambda_k}[\log\beta_{kw}])"/>
is the log-partition function.
The solutions for
<img src="https://latex.codecogs.com/svg.image?\inline&space;\gamma_{dk}=\alpha+\sum_wn_{dw}\phi_{dwk}"/>
and
<img src="https://latex.codecogs.com/svg.image?\inline&space;\lambda_{kw}=\eta+\sum_dn_{dw}\phi_{dwk}"/>
come from direct gradient calculation.
After substituting the optimal solutions, we compute the marginal likelihood by adding the three terms, which are all connected to (the negative of) Kullback-Leibler divergence.

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
 * Set `0<rho<1` for online learning with partial_fit.
 * Set `mult["doc"]=100` or `mult["word"]=100` or some large value to disable the corresponding Bayesian priors.

References
---

1. Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
Dirichlet Allocation. Advances in Neural Information Processing Systems 23
(NIPS 2010).
2. Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
