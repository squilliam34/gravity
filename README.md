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

Due to the large differences in scales of companies and their valuations, I chose to transform the market caps with a log transformation to compress the space in which market capitalization exists. Mechanically, if this space wasn't compressed, the gravity measure would be dominated by a few mega-cap companies, and my distance measurements would have little measurable effect.

### Distance
Deviating from the gravity model used for trade, I didn't want to use geographic locations to represent physical distance. For one, geographic distance is less and less of a barrier in today's world of multinational corporations and international supply chains. Secondly, distances between where companies are based doesn't tell that much. For example, both Bank of America and Honeywell are headquartered in Charlotte, but they fall within very different industries and as such, shouldn't be grouped together. As a result, I propose the following distance metrics: semantic similarity and differences in factors.

In order to combine them, I used a time-varying weighted average between the 2: $\lambda * D_{factor} + (1 - \lambda) * D_{Similarity}$

#### Semantic Similarity
Measuring distance by semantic similarity uses an NLP (Natural Language Processing) approach. For each stock in our universe, retrieve a description of the company and embed it into a vector of numerical values. Then, the distance between companies $i$ and $j$ can be found by calculating the cosine similarity, which takes the cosine of the angle between the 2 vectors, and subtracting it from 1. The idea is that the cosine of the angle between two vectors that are closer together/more similar will be closer to 1 so subtracting it from 1 will produce a smaller "distance". Similarly, the cosine of the angle between two vectors that are farther apart/less similar will be closer to 0 so subtracting it from 1 will produce a larger "distance".

##### Business Descriptions
I had originally hoped to use 10-K business descriptions to produce more robust results. However, this was both computationally expensive due to their large size and proved to be challenging since 10-Ks are not formatted consistently, which made extracting the business description problematic -- even through a variety of web scraping approaches and API services. As a result, I decided to use company descriptions pulled from Yahoo Finance due to them offering consistent company descriptions that were available across their universe of coverage.

##### Model Selection
I experimented with various different HuggingFace models for embedding text. I originally used a general and lightweight model `sentence-transformers/all-MiniLM-L6-v2`. Despite its speed and efficiency of performance, the model seemed to be unable to capture more nuanced relationships when examining various distance scores. I then tried a more specialized model `FinLang/finance-embeddings-investopedia` in hopes that its specialized finance corpus would provide better distinction. However, in evaluating the results, it seemed overspecialize in finance and didn't differentiate between sectors as well. Finally, I settled on `sentence-transformers/all-mpnet-base-v2`. This model takes longer to run and requires more computing power, but it seemed best equipped to capture nuance between companies while still upholding fundamental relationships.

#### Differences in Factors
Aside from capturing structural differences, I also wanted to capture differences in price behavior as another dimension of similarity. To do this, I explored the idea of specifying a parsimonious multi-factor model inspired by empirical asset pricing models[^4]. The specification made use of market exposure, interest rate sensitivity, and momentum in the spirit of the market, style, and macro factors that are employed by institutional risk models like Barra, designed to capture major differences in macro sensitivity and trend-following behavior. Daily percentage returns of the S&P500 served as the market exposure factor. Daily absolute change in the 10 year treasury yield offered interest rate sensitivity. The momentum factor utilized a 12-1 month return spread in the spirit of the $WML$ factor used in the Carhart four-factor model[^5]. To implement this, at each point in the sample, I calculated the returns over the past 12 months (252 trading days) and omitted the most recent month (the last 21 trading days). Then for each day, returns across active stocks would be ranked, and the spread between the average returns of the first and last decile was calculated to be used as a momentum factor.

Due to the volume of data that I have access to, I chose to run a rolling model, where factors were recomputed every month using the previous 252 trading days (about a year of trading) since exposure to these factors isn't static. 

Behavioral distance is then measured by differences in stocks' exposures to those common risk factors, similarly to multi-factor risk models. When taking the distance, I employed the Mahalanobis distance due to the fact that it accounted for scale and correlation between factors. 

However, after calculating distances, they were on a different scale from the cosine distances. To avoid having cosine distances be overshadowed, I transformed the factor distances using $1-e^{-x}$, as this helped to compress the distance space and also bounded distances within the range (0, 1).

#### Lambda
For Lambda, I wanted a measure that would vary over time and dynamically weight each distance factor depending on market conditions. This offered a more robust weighting system than arbitrarily weighting each factor. To do so, I chose to calculate $\lambda$ using the $VIX$, as my rationale was that during more volatile periods, the price behavior -- and therefore factor difference -- would be more important than structural differences between companies. However, the $VIX$ as itself does not neatly work as a weight since it scales from 0 to technically infinity. Additionally, if we treat the long term average of the $VIX$ as a baseline market regime where each distance factor should be weighted more or less evenly, the long term average of ~19 - ~21 doesn't offer convenient weighting.

To address these issues, I passed the $VIX$ value through a sigmoid function of the form $\frac{1}{1 + e^{-k * \frac{VIX}{\text{threshold}}}}$. Here, $k$ works as a measure of how steep the sigmoid function is and how reactive $\lambda$ is to a change in the $VIX$. $\text{threshold}$ serves to scale the $VIX$ down, based on how far away from the long term average the current measure is.

[^1]: Tinbergen, J. (1962). Shaping the world economy: Suggestions for an international economic policy. Twentieth Century Fund.

[^2]: Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. The Review of Financial Studies, 20(4), 1113–1138. https://doi.org/10.1093/revfin/hhm003.

[^3]: Gabaix, X. (2011). The granular origins of aggregate fluctuations. Econometrica, 79(3), 733–772. https://doi.org/10.3982/ECTA8769.

[^4]: Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56. https://doi.org/10.1016/0304-405X(93)90023-5.

[^5]: Carhart, M. M. (1997). On persistence in mutual fund performance. The Journal of Finance, 52(1), 57-82. https://doi.org/10.1111/j.1540-6261.1997.tb03808.x