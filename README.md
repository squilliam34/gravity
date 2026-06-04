# Gravity Model For Stocks

This project looks to adapt the gravity model of trade[^1], frequently used in economic trade literature, to analyze stocks.

## Gravity For Trade
The gravity model of trade is a well known economic framework for modeling trade flows between 2 countries that adopts the standard formula for gravity $G = \frac{M_1 M_2}{D^2}$. It models trade flows using respective $GDP$ of each country as their mass and the physical distance $D$:
$\\ Trade_{ij} = A \times \frac{GDP_i \times GDP_j}{D_{ij}}$, where $A$ is a constant.

## Adoption
Similarly, this project will test whether a gravity model can be used as a more robust measure of correlation between stocks.

### Mass
The obvious choice for the "mass" of a stock is to use each stock's market capitalization. The other justification for using market cap as mass is that companies with larger market caps exhibit greater influence on their index and on similar constituents with a smaller market cap. 

Both the NASDAQ and S&P indices are market cap weighted indices, supporting the argument that larger companies exhibit greater "gravitational pull", or influence, on their index.

Additionally, there is empirical evidence that larger companies serve as leading indicators of returns of smaller firms, especially within the same industry[^2]. Furthermore, aside from effecting smaller companies, there is also empirical evidence that larger companies have a greater effect on the economy as well[^3], further supporting the idea of large cap stocks exhibiting more "force".

### Distance

[^1]: Tinbergen, J. (1962). Shaping the world economy: Suggestions for an international economic policy. Twentieth Century Fund.s

[^2]: Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. The Review of Financial Studies, 20(4), 1113–1138. https://doi.org/10.1093/revfin/hhm003 

[^3]: Gabaix, X. (2011). The granular origins of aggregate fluctuations. Econometrica, 79(3), 733–772. https://doi.org/10.3982/ECTA8769