Background:
The business was seeing an uptick of instances where brokerage customer accounts were hacked, and their funds were used to pump up the stock price of stocks.
These unauthorized transactions lead to financial losses for the brokerage.

Business Goal:
Place systematic blocks on these securities to not allow trading of equities most likely to be used for pump and dump stock price manipulation.

Data Provided:
The business provided a list of customers (~20) over a 5 month period, whose accounts were taken over and whose funds were used to pump stock prices.
There isn't a systematic way to flag these customers and the underlying security currently.

Considerations:
There were some considerations that had to be made with the given problem:
- 20 is a relatively small sample of True Positives (TP) to train and test a model
- Features were limited to security related data (ie currency, country of origination, various price details, date of origination, market cap) and customer demographics and transactions 
- Balance False Positives (FP) and False Negatives (FN). FPs would lead to blocking securities unnecessarily, while FN would mean potentially failing to stop fraudulent activity

Approach:
Curating the Model Training and Testing Dataset






Next Steps
- Given fraudulent transactions generally occur through digital platforms (fraudsters generally do not call in to place trades)