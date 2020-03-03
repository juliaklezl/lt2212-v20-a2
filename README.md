# LT2212 V20 Assignment 2

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

1) I tokenize the words by splitting them at empty spaces. In order to filter
out useless data, I remove all punctuation and numbers and lowercase all 
tokens. I only include words which occur more than 10 times in the total
corpus in order to keep the amount of features manageable for part 3, and to 
exclude "words" without real meaning, such as the email-addresses/IDs in the 
beginning of each posting. 

2) I chose the PCA  implementation of sklearn. For the bonus part, I ran the
program again using TruncatedSVC.

3) Model #1 is a Decision Tree Classifier with default settings, Model #2 is 
Gaussian Naive Bayes

4) I trained and evaluated both classifiers with the full feature set, with
approximately half the feature set (6000 features), approx. 25% (3000 features),
approx. 10% (1300 features), and 5% (600 features). THe results are as follows:
The reported results are accuracy, weighted average of precision, recall, and
f1-score (if only one value is given, the numbers were identical). 

          Decision Tree  Gaussian NB
unreduced 0.56		 0.74
6000 f.   0.14		 0.12, 0.23, 0.12, 0.09
3000 f.   0.14		 0.12, 0.25, 0.12, 0.10
1300 f.   0.16		 0.14, 0.25, 0.14, 0.12
600 f.    0.15		 0.15, 0.26, 0.15, 0.13

I am surprised how quickly the performance of both classifiers drops with the
reduced dimensions. I would have expected at least the 50% - version to still
yield sensible results. Therefore, it seems that this feature set could not 
successfully be merged into meaningful combined features. It is also interesting that
the Gaussian NB classifier performs far better than the Decision Tree in the unreduced
version, but has a very similar performance in the reduced versions.
Another thing I find interesting is the fact that the different measurements were all
rounded to the same numbers in the Decision Tree classifier, but differed significantly
in the Gaussian NB (especially the precision regularly scores around 10 percentage 
points higher than the other measurements). This might be a coincidence, but could 
possibly also mean that the Decision Tree's results are more balanced/stable.

Part Bonus
For the bonus part, I implemented the same program and classifiers, but used sklearn's
Truncated SVC algorithm for the dimensionality reduction instead. 
          Decision Tree  Gaussian NB
unreduced 0.56           0.74, 0.74, 0.73, 0.73
6000 f.   0.15           0.12, 0.24, 0.12, 0.10
3000 f.   0.14           0.12, 0.24, 0.12, 0.10
1300 f.   0.16           0.14, 0.27, 0.14, 0.13
600 f.    0.15           0.16, 0.27, 0.16, 0.14

Using another algorithm for the dimensionality reduction did not have a significant 
effect on the results.
