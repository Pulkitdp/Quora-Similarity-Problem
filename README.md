# Quora-Similarity-Problem

Problem statement

To predict which of the provided pairs of questions contain two questions with the same meaning.
Source

Kaggle: Quora question pair
Focus area

To achieve a probability of a pair of questions to be duplicates so that you can choose any threshold of choice with minimal misclassification.
Data source

The data is available on Kaggle, features of which are briefly summarised here -

    id - the id of a training set question pair
    qid1, qid2 - unique ids of each question (only available in train.csv)
    question1, question2 - the full text of each question
    is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

