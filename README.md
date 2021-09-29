# AI-Classification-Algorithm
For this coursework, we were given a set of emails and were required to code a classifier in Python that determined whether an email was spam or 'ham' by calculating the likelihood that a set of spam words appears in each email. I was able to achieve an accuracy of 80% correctness using logarithms and conditional likelihoods. 


-----------------------------------
----- CLASSIFICATION | README -----
----------------------------------- 

This README is on Part 1. I did not implement Part 2.

----- INTRODUCTION -----

This coursework was based on supervised machine learning. The aim (for Part 1) was to create a classifier to detect whether a set of email messages was either spam or ham. A training dataset (1000 rows) and a testing dataset (500 rows) were both provided, each with a row of response variables/class labels (0 for ham and 1 for spam). There were 54 keywords that could be present in the dataset, and the classifier had to differentiate between spam and ham based on the probabilities that each keyword would appear in a certain class.

The method I used to implement the classifier was a Naive Bayes model, a simple probabilistic classifier. Naive Bayes classifiers assume that a given feature's value is independent of the value of any other feature, given the class variable.

An advantage of naive Bayes is that it only requires a small number of training data to estimate the parameters necessary for classification. (https://en.wikipedia.org/wiki/Naive_Bayes_classifier)


----- IMPLEMENTATION -----

My general implementation involved first building a model to find the probability that a given message belonged to a certain class, p(C = c| message), known as the class priors. To calculate the class priors, I found the total number of labels for each class (in other words, how many emails of class 0 and class 1 existed in the training data). After finding the totals, I then divided these by the total number of emails there were in the dataset. This gave the proportion of each class in the dataset.

Following on, the next step was to calculate the actual probability of the message belonging to a certain class, called the class conditional likelihood. As this was a Naive Bayes model, I assumed that there was a conditional independence of keywords given a class, and that there was a multinomial distribution for each class. I then counted the relative frequencies of each keyword in the training data - how many times the feature appeared out of the total number of all present keywords in that class.

I then calculated the multinomial distribution for each feature in each class. To reduce the problem of zero probability, I used Laplace smoothing with alpha = 1. The formula I used was to calculate the multinomial distribution was:

(number of times keyword appears + alpha) / (total keywords present in class + number of different keywords * alpha)

Before I went further, I computed the logarithms of each value in the class priors and conditional likelihood arrays. This was to increase the 'numerical stability' of the algorithm so that the algorithm does not process numbers close to 0 (which would make it more error-prone) but instead processes values from negative infinity to 0. Taking logarithms does not affect the overall distribution as argmax(log(f(x))) is equivalent to argmax(f(x)).

I then defined a method called 'train' which simply calculated and stored the logarithm of the class priors and the logarithm of the class conditional likelihoods in instance variable numpy arrays.

Lastly, I defined a method called 'predict' that would use the trained class priors and conditional likelihoods on new/test data. For each feature in a row, the conditional likelihood value (for both classes) were added to a total for that class. The class prior was then added to the total for the corresponding class. Each of these new calculated values was then appended to a simple list.

I computed the argmax() on the above list to find the index where the highest probability was positioned. This index was then appended to a numpy array with indexes for every other email in the dataset. In this case, the resulting numpy array only contained 0s and 1s, as there were only two classes. Therefore, the formula I used to classify the emails was:

argmax[log(p(C = c)) + ∑ (each feature) log(class, feature)]


----- TESTING -----

When testing the classifier, I obtained an accuracy of:

Training Set: 0.804 / 80.4% (1000 rows)
Testing Set: 0.818 / 81.8% (500 rows)

From these results, it can be inferred that my implemented algorithm returns an average accuracy of about 81.1%. An accuracy in the 80th percentile is significantly better than a random classification (which would be 50% in this case).

To improve my classifier, I could refine the features selected from the data to make them more specific and accurate. Additionally, I could also add in more data with more keywords present to make the class priors and conditionals more precise. Overall, my classifier is quite accurate and can differentiate between spam and ham email messages to a suitable degree.


