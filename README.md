# TextUnderstanding_MultiLabel

Understand news articles and predict the probability that they contain specific topics. In essence, it is a Multi Label regression problem.

I used a Bert model with additional layers with an output number of neurons equal to the number of different topics. I also used sigmoid activation functions in each final neuron because the percentages don't sum up to one necessarely.


