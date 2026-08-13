# Gravity Model For Stocks

This project looks to adapt the gravity model of trade[^1], frequently used in economic trade literature, to analyze stocks.

## Gravity For Trade
The gravity model of trade is a well known economic framework for modeling trade flows between 2 countries that adopts the standard formula for gravity $G = \frac{M_1 M_2}{D^2}$. It models trade flows using the respective $GDP$ of each country as their mass and the physical distance $D$:
$\\ Trade_{ij} = A \times \frac{GDP_i \times GDP_j}{D_{ij}}$, where $A$ is a constant.

## Adoption For Stocks
Similarly, this project will test whether a gravity model can be used to model networks within the stock market.

### Mass
The obvious choice for the "mass" of a stock is to use each stock's market capitalization. The other justification for using market cap as mass is that companies with larger market caps exhibit greater influence on their index and on similar constituents with a smaller market cap. 

Both the NASDAQ and S&P indices are market cap weighted indices, supporting the argument that larger companies exhibit greater "gravitational pull", or influence, on their index.

Additionally, there is empirical evidence that larger companies serve as leading indicators of returns of smaller firms, especially within the same industry[^2], supporting the notion that larger companies influence smaller companies. Furthermore, aside from effecting smaller companies, there is also empirical evidence that larger companies have a greater effect on the economy as well[^3], as a handful of major corporations are accountable for a disproportionately large share of economic output. Therefore, when these large firms experience major shocks, the effects are too big to be "averaged out" and are propagated throughout the economy, further supporting the idea of large cap stocks exhibiting more "force".

Due to the large differences in scales of companies and their valuations, I chose to transform the market caps with a log transformation to compress the space in which market capitalization exists. Mechanically, if this space wasn't compressed, the gravity measure would be dominated by a few mega-cap companies, and my distance measurements would have little measurable effect.

### Distance
Deviating from the gravity model used for trade, I didn't want to use geographic locations to represent physical distance. For one, geographic distance is less and less of a barrier in today's world of multinational corporations and international supply chains. Secondly, distances between where companies are based doesn't tell that much. For example, both Bank of America and Honeywell are headquartered in Charlotte, but they fall within very different industries and as such, shouldn't be grouped together. As a result, I propose the following distance metrics: semantic similarity and differences in factors.

In order to combine them, I took the product between each distance measure. This ensured that 2 companies were required to  behave similarly and display structural similarity to have a small distance between them.

#### Semantic Similarity
Measuring distance by semantic similarity uses an NLP (Natural Language Processing) approach. For each stock in our universe, retrieve a description of the company and embed it into a vector of numerical values. Then, the distance between companies $i$ and $j$ can be found by calculating the cosine similarity, which takes the cosine of the angle between the 2 vectors, and subtracting it from 1. The idea is that the cosine of the angle between two vectors that are closer together/more similar will be closer to 1 so subtracting it from 1 will produce a smaller "distance". Similarly, the cosine of the angle between two vectors that are farther apart/less similar will be closer to 0 so subtracting it from 1 will produce a larger "distance".

##### Business Descriptions
To generate the most robust results, I wrote a script that would scrape the "Business Overview" from a company's annual 10-K filing. Due to companies businesses changing over time, I pulled these at different points in time: 2010, 2015, 2020, 2025. This enabled me to capture time varying changes in business models. 

In some cases, my script was not able to successfully extract the "Business Overview", particularly in earlier years, so I manually filled them in from the `sec.gov` website.

##### Model Selection
Due to the potentially large context windows required to embed the "Business Overview" for a given company, I found that using a frontier model for embedding worked the best. Research online pointed me in the direction of the `Gemini` API for embedding. Not only is it known to perform well for semantic similarity tasks, but it also is typically cheaper than other frontier models.  

#### Differences in Factors
Aside from capturing structural differences, I also wanted to capture differences in price behavior as another dimension of similarity. To do this, I explored the idea of specifying a parsimonious multi-factor model inspired by empirical asset pricing models[^4]. The specification made use of market exposure, interest rate sensitivity, and momentum in the spirit of the market, style, and macro factors that are employed by institutional risk models like Barra, designed to capture major differences in macro sensitivity and trend-following behavior. Daily percentage returns of the S&P500 served as the market exposure factor. Daily absolute change in the 10 year treasury yield offered interest rate sensitivity. The momentum factor utilized a 12-1 month return spread in the spirit of the $WML$ factor used in the Carhart four-factor model[^5]. To implement this, at each point in the sample, I calculated the returns over the past 12 months (252 trading days) and omitted the most recent month (the last 21 trading days). Then for each day, returns across active stocks would be ranked, and the spread between the average returns of the first and last decile was calculated to be used as a momentum factor.

Due to the volume of data that I have access to, I chose to run a rolling model, where factors were recomputed every month using the previous 252 trading days (about a year of trading) since exposure to these factors isn't static. 

Behavioral distance is then measured by differences in stocks' exposures to those common risk factors, similarly to multi-factor risk models. When taking the distance, I employed the Mahalanobis distance due to the fact that it accounted for scale and correlation between factors. 

However, after calculating distances, they were on a different scale from the cosine distances. To avoid having cosine distances be overshadowed, I transformed the factor distances using $1-e^{-x}$, as this helped to compress the distance space and also bounded distances within the range (0, 1).

## Time-Varying Networks
I reconstructed clusters every 5 years, in accordance with my business overviews. This allowed me to capture differences that may have arisen from changing business models. For example, `AMZN` is a very different company with their current cloud computing business than they were in their early e-commerce days. 

### Constituents
Determining constituents proved to be a problem. I could not simply use today's S&P 500 constituents, as they aren't representative of the S&P 500 from 15 years ago. Not to mention that in my later attempts at portfolio construction, using current constituents would introduce survivorship bias, as only the companies that are still publicly traded are available. 

### Clustering
For my clustering, due to the high dimensional nature of the embeddings, I need to employ some form of dimensionality reduction in order to avoid the curse of dimensionality. I chose UMAP for this process since it helps to ensure separation in points by grouping close points closer and far points farther. It also is generally thought to preserve the global shape of the data well. 

After applying UMAP to my data, I employed Hierarchicial Density based Spatial Clustering of Applications with Noise (HDBSCAN), which I chose for several reasons. For one, it is non-parametric, which was very important to me. The market landscape can change drastically across periods, and relying on a finely tuned clustering algorithm could mean economically meaningful clusters in one year, and nonsensical clusters in another. Secondly, it also manages data with varying levels of density well with its hierarchical structure. This is important since some clusters might be tighter grouped together than others. Finally, I could enforce cluster membership for all points by assigning points that were classified as noise to the cluster that HDBSCAN gave as the highest probability grouping.

### Intra-Cluster Gravity
After clustering, I chose to calculate "gravity" within each cluster and form networks. This was primarily motivated by the idea that within an industry (or cluster in our case), a larger company can influence smaller companies [^2]. Therefore, by calculating gravity within a cluster, I ideally would have been able to better isolate the "gravitational" relationship between alike companies.

## Portfolio Construction
To test whether the gravity networks contained useful information for portfolio construction, I used them to identify a “leader” within each cluster. For every company, I calculated its total gravitational connection to the other companies in its cluster. A firm with high within-cluster gravity is both economically significant and closely connected to its peers through the structural and factor-based dimensions captured by the model.

My intuition was that these highly central firms could serve as representatives of their respective clusters. Because a cluster leader is strongly connected to many of the companies around it, its returns may reflect the cluster’s shared economic exposures more closely than those of a peripheral firm, whose performance may be driven by more idiosyncratic factors. Selecting one leader from each cluster therefore creates a portfolio with exposure to the market’s different underlying groups while limiting redundancy among companies with similar characteristics.

Here, “leader” does not necessarily mean that the company causally drives the returns of its peers. It refers more narrowly to the company occupying the most central position in the gravity network. The primary hypothesis is not necessarily that cluster leaders will outperform the market, but that they can provide a compressed representation of it. If each leader successfully reflects the shared exposures of its cluster, a portfolio containing relatively few leaders may reproduce much of the S&P 500’s behavior. Evidence in favor of this hypothesis would include low tracking error, similar market beta, comparable volatility and drawdowns, and stable performance across market regimes.

The appropriate weighting scheme depends on the question being tested. An equal-weighted portfolio gives every cluster equal importance, while a market-cap-weighted portfolio concentrates exposure in the largest selected leaders. A third approach assigns each leader the aggregate S&P 500 weight of the cluster it represents. This cluster-weighted construction provides the most direct test of whether a single central firm can serve as a proxy for its broader group.

### Equal Weighting

[^1]: Tinbergen, J. (1962). Shaping the world economy: Suggestions for an international economic policy. Twentieth Century Fund.

[^2]: Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. The Review of Financial Studies, 20(4), 1113–1138. https://doi.org/10.1093/revfin/hhm003.

[^3]: Gabaix, X. (2011). The granular origins of aggregate fluctuations. Econometrica, 79(3), 733–772. https://doi.org/10.3982/ECTA8769.

[^4]: Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56. https://doi.org/10.1016/0304-405X(93)90023-5.

[^5]: Carhart, M. M. (1997). On persistence in mutual fund performance. The Journal of Finance, 52(1), 57-82. https://doi.org/10.1111/j.1540-6261.1997.tb03808.x