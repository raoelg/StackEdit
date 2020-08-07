

# Random Indexing

Random indexing involves a simple process for creating vector embeddings for words and other tokens.

It starts with a very large sample of examples of *contexts* within the tokens occur. For instance, a large sample of sentences—the contexts—that contain the words for which the vector embeddings have to be derived. The following sequence of steps:

1. Generate random vectors $v_i \in \mathbb{R}^n$ for each context $i$. A common value for $n$ is 300. Most components in $v_i$ are 0. Only a small percentage (e.g., 10%) is randomly assigned an equal number of 1's and -1's.
2. The vector embeddings for each token $t$ of interest as computed by adding the context vectors of each context in which $t$ occurs for each occurence. That is, $r_t = \sum_{i \in \{i|t\in C_i\}} v_i$


## Implementation in R

The following is a simple implementation in R. A text file was obtained that contains more than 170,000 text fragments (sentences?) from a Wikipedia dump.

```r
library(Matrix)
library(tidyverse)

text8l = readLines(unz("~/Desktop/text8l.zip", "text8l")) # 170,052 lines of text

# Build context-token data frame
wiki = tibble(text = text8l) %>% 
	rownames_to_column("context") %>% 
	mutate(context = as.integer(context)) %>% 
	unnest_tokens(token, text) %>% 
	add_count(token) # for filtering purposes
	
# Create sparse matrix of context vectors
set.seed(10)
idx = replicate(length(text8l), sample(300,30))
idx = cbind(c(col(idx)), c(idx))
contextVectors = Matrix(0, nrow = length(text8l), ncol=300)
contextVectors[idx] = c(-1L,1L)

rivecs_10up = wiki %>% 
	filter(n>9) %>%  # reduces dictionary to ~ 50,000 words
	group_by(token) %>% 
	summarise(vec = list(as.integer(colSums(contextVectors[context,,]))))

RIVECS_10UP = do.call(rbind, rivecs_10up$vec)
rownames(RIVECS_10UP) = rivecs_10up$token
```

The matrix `RIVECS_10UP` contains the 300-dimensional word vectors (rowwise) for 47,134 words.

## Statistical properties 

What are the statistical properties of these word vectors? How do they encode relations between words?

### Mean

We consider $E(r_t)$ and $E({\bf 1}'r_t)$, but since $E({\bf 1}'r_t) = {\bf 1}'E(r_t)$ the former suffices. Since $r_t = \sum_{i\in\{k:t\,\in\,C_k\}}n_{it}v_i,$
we have $$E(r_t) = \sum_{i\in\{k:t\in C_k\}}n_{it}E(v_i) = \sum_{i\in\{i:k\in C_k\}}n_{it}\cdot {\bf 0} = {\bf 0}.$$

### (Co)Variance

We consider the covariance matrix that is composed of $E(r_tr_t')$ and $E(r_tr_s')$, and we consider the inner product of the vector embeddings, $E(r_t'r_s)$. Since $E(r_t r_t')$ is a special case of $E(r_t r_s')$, and $E(r_t'r_s) = \mathrm{tr}\{E(r_t r_s')\}$, $E(r_t r_s')$ suffices.

$$ E(r_t r_s') = 
E\left[\big(\sum_{i\in\{k:t\in C_k\}}n_{it}v_i\bigg) \bigg(\sum_{j\in\{k:s\in C_k\}}n_{js}v_j\bigg)'\right] \\ 
= \sum_{i\in\{k:t\in C_k\}} \sum_{j\in\{k:t\in C_k\}} n_{it}n_{js} E(v_iv_j') 
= \sum_{i\in\{k:t,s\in C_k\}} n_{it}n_{is}E(v_iv_i')
$$

Note that $v_iv_i' = \sum_{i \in R} e_i e_i'$ where $e_i$ is the $i$-the column in the identity matrix $I_n$, and $R$ is a random sample of $m$ indices, uniformly drawn from $\{1, 2,\ldots, n\}$ without replacement. Hence, $E(v_i v_i') = {m\over n}I_n$, and

$$ 
E(r_t r_s') = {m\over n}I_n\cdot \sum_{i\in \{k:t,s\in C_k\}} n_{it}n_{is}
= {m\over n}I_n\cdot \sum_{i=1}^N n_{it}n_{is},$$

because $n_{it} = 0$ if $t\not\in C_i$. The conclusion is that, $E(r_t r_s')$ encodes the co-occurence of tokens $t$ and $s$. If $t=s$, we have 

$$
E(r_tr_t') = {m\over n} I_n \cdot \sum_{i=1}^N n_{it}^2,
$$

which encodes the relativel frequency of token $t$. Furthermore, the inner product of any pair of word vector embeddings is

$$
E(r_t'r_s) = m\cdot \sum_{i=1}^N n_{it}n_{is} = m\cdot \mathbf{n}_t'\mathbf{n}_s,
$$

the inner product of the $N$ dimensional vectors of token-context counts. As a consequence, 

$$
{E(r_t'r_s) \over \sqrt{E(r_t'r_t)E(r_s'r_s})} = {\mathbf{n}_t\cdot \mathbf{n}_s \over \parallel\mathbf{n}_t\parallel \parallel\mathbf{n}_s\parallel},
$$

the cosine angle between the vectors of token-context counts. For large enough $N$, 

$${r_t\cdot r_s \over \parallel r_t\parallel \parallel r_s\parallel} \approx {\mathbf{n}_t\cdot \mathbf{n}_s \over \parallel\mathbf{n}_t\parallel \parallel\mathbf{n}_s\parallel}.$$

More formally,

$$ \lim_{N\rightarrow \infty} \bigg( {r_t\cdot r_s \over \parallel r_t \parallel \parallel r_s \parallel} - {\mathbf{n}_t\cdot \mathbf{n}_s \over \parallel \mathbf{n}_t \parallel \parallel \mathbf{n}_s \parallel} \bigg) = 0. $$


### Distribution

Because the $r_t$'s are sums of identically distributed random variables with finite variance a CLT can be invoked to conclude that  $r_t \stackrel{A}{\sim} N({\bf 0},{m\over n}\sum_i n_{it}^2\cdot I_n)$.

## Other word vector embeddings

A number of different approaches to finding vector embeddings exist. Well known are

- GloVe
- word2vec
- Fasttext
- Elmo
- BERT

With the exception of GloVe, these are all learned by training neural networks to predict the presence of (combinations of) the target token from it's context. GloVe vectors are obtained by least squares methods and essentially boils down to classical multidimensional scaling: A set of location vectors in n-dimensional space are found that reproduces the observed distances between cases as well as possible.

The advantage of random indexing is that it is easy to extend the dictionary of tokens: All the other approaches require a complete retraining of the vectors. Random indexing only needs the existing context vectors for each token added, and word embeddings can be easily updated with new contexts.

<!--stackedit_data:
eyJwcm9wZXJ0aWVzIjoidGl0bGU6IFJhbmRvbSBJbmRleGluZy
B2ZWN0b3JzXG5hdXRob3I6IFJhb3VsIEdyYXNtYW5cbmRhdGU6
ICdTdW4gT2N0IDIwIDIwMTkgMTM6MjQ6MDAgR01UKzAyMDAgKE
NlbnRyYWwgRXVyb3BlYW4gU3VtbWVyIFRpbWUpJ1xudGFnczog
J3JhbmRvbSBpbmRleGluZywgd29yZCB2ZWN0b3JzLCBlbWJlZG
RpbmdzJ1xuIiwiaGlzdG9yeSI6Wy0xMTg3NzQ4MTY5LDEzMjcz
NDEyMDZdfQ==
-->