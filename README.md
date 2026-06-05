# Gravity Model For Stocks

This project looks to adapt the gravity model of trade[^1], frequently used in economic trade literature, to analyze stocks.

## Gravity For Trade
The gravity model of trade is a well known economic framework for modeling trade flows between 2 countries that adopts the standard formula for gravity $G = \frac{M_1 M_2}{D^2}$. It models trade flows using the respective $GDP$ of each country as their mass and the physical distance $D$:
$\\ Trade_{ij} = A \times \frac{GDP_i \times GDP_j}{D_{ij}}$, where $A$ is a constant.

## Adoption For Stocks
Similarly, this project will test whether a gravity model can be used as a more robust measure of correlation between stocks.

### Mass
The obvious choice for the "mass" of a stock is to use each stock's market capitalization. The other justification for using market cap as mass is that companies with larger market caps exhibit greater influence on their index and on similar constituents with a smaller market cap. 

Both the NASDAQ and S&P indices are market cap weighted indices, supporting the argument that larger companies exhibit greater "gravitational pull", or influence, on their index.

Additionally, there is empirical evidence that larger companies serve as leading indicators of returns of smaller firms, especially within the same industry[^2], supporting the notion that larger companies influence smaller companies. Furthermore, aside from effecting smaller companies, there is also empirical evidence that larger companies have a greater effect on the economy as well[^3], as a handful of major corporations are accountable for a disproportionately large share of economic output. Therefore, when these large firms experience major shocks, the effects are too big to be "averaged out" and are propagated throughout the economy, further supporting the idea of large cap stocks exhibiting more "force".

### Distance
Deviating from the gravity model used for trade, I didn't want to use geographic locations to represent physical distance. For one, geographic distance is less and less of a barrier in today's world of multinational corporations and international supply chains. Secondly, distances between where companies are based doesn't tell that much. For example, both Bank of America and Honeywell are headquartered in Charlotte, but they fall within very different industries and as such, shouldn't be grouped together. As a result, I propose the following distance metrics: semantic similarity and differences in factors.

#### Semantic Similarity
Measuring distance by semantic similarity uses an NLP (Natural Language Processing) approach. For each stock in our universe, retrieve a description of the company and embed it into a vector of numerical values. Then, the distance between companies $i$ and $j$ can be found by calculating the cosine similarity, $\cos \theta_{ij} = \frac{\vec{v}_i \cdot \vec{v}_j}{\lVert \vec{v}_i \rVert \lVert \vec{v}_j \rVert}$, which takes the cosine of the angle between the 2 vectors, and subtracting it from 1. The idea is that the cosine of the angle between two vectors that are closer together/more similar will be closer to 1 so subtracting it from 1 will produce a smaller "distance". Similarly, the cosine of the angle between two vectors that are farther apart/less similar will be closer to 0 so subtracting it from 1 will produce a larger "distance".

##### Business Descriptions
I had originally hoped to use 10-K business descriptions to produce more robust results. However, this was both computationally expensive due to their large size and proved to be challenging since 10-Ks are not formatted consistently, which made extracting the business description problematic -- even through a variety of web scraping approaches and API services. As a result, I decided to use company descriptions pulled from Yahoo Finance due to them offering consistent company descriptions that were available across their universe of coverage.

##### Model Selection
I experimented with various different HuggingFace models for embedding text. I originally used a general and lightweight model `sentence-transformers/all-MiniLM-L6-v2`. Despite its speed and efficiency of performance, the model seemed to be unable to capture more nuanced relationships when examining various distance scores. I then tried a more specialized model `FinLang/finance-embeddings-investopedia` in hopes that its specialized finance corpus would provide better distinction. However, in evaluating the results, it seemed overspecialize in finance and didn't differentiate between sectors as well. Finally, I settled on `sentence-transformers/all-mpnet-base-v2`. This model takes longer to run and requires more computing power, but it seemed best equipped to capture nuance between companies while still upholding fundamental relationships.

#### Differences in Factors

[^1]: Tinbergen, J. (1962). Shaping the world economy: Suggestions for an international economic policy. Twentieth Century Fund.

[^2]: Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. The Review of Financial Studies, 20(4), 1113–1138. https://doi.org/10.1093/revfin/hhm003 

[^3]: Gabaix, X. (2011). The granular origins of aggregate fluctuations. Econometrica, 79(3), 733–772. https://doi.org/10.3982/ECTA8769