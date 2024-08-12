#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Sukhum Boondecharak
# #### Student ID: S3940976
# 
# Date: 04 Oct 2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# The objective for task 2 is to create feature representations for job advertisement descriptions. These representations will be used to capture the essential information within the text data, making it suitable for machine learning models. The task involves two main feature generation processes:
# 
# 1. Bag-of-Words Model: This approach involves creating count vector representations for each job advertisement description based on a preprocessed vocabulary. The generated count vectors represent the frequency of each word in the descriptions.
# 2. Word Embeddings: This is to capture semantic relationships between words and can provide rich representations for text data. In this sub-task, I chose FastText as a word embedding model and initially created both unweighted and TF-IDF weighted vector representations for job advertisement descriptions.
# 
# Task 3 focuses on building machine learning models to classify job advertisements into specific categories based on their textual content. The primary goal is to investigate two key questions:
# 
# - Q1: Which language model, among those created in Task 2, performs best when combined with chosen machine learning models? Various models will be built based on different feature representations, and their performance will be evaluated.
# 
# - Q2: Does more information improve accuracy? Different combinations of features will be explored, including using only the job title, only the job description, or both. By experimenting with these combinations, We aim to understand whether incorporating additional information, such as job titles, improves the accuracy of the classification models.

# ## Importing libraries 
# 
# Various libraries are imported for different activities.

# In[1]:


import os
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from collections import Counter


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ### Bag-of-Words (BoW) Model: Count Vectors
# 
# First, generate the Count vector representation for each job advertisement description using the vocabulary created in Task 1. The count vectors will be combined and saved at the end of this task.

# In[2]:


# Load the cleaned data
with open("cleaned_descriptions.txt", "r", encoding="utf-8") as file:
    cleaned_descriptions = file.readlines()

# Load the vocabulary
with open("vocab.txt", "r") as file:
    vocab = [line.strip().split(":")[0] for line in file]

# Indicate original data folder
original_data_folder = "data"

# Initiate an empty list to store webindex numbers
webindex_numbers = []

# Iterate through the original data files and extract webindex
for category_folder in os.listdir(original_data_folder):
    category_path = os.path.join(original_data_folder, category_folder)
    if os.path.isdir(category_path):
        for job_file in os.listdir(category_path):
            if job_file.startswith("Job_") and job_file.endswith(".txt"):
                with open(os.path.join(category_path, job_file), "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Extract the webindex from the original data and remove the newline character
                    webindex = content.split("Webindex: ")[1].split("\n")[0]
                    
                    webindex_numbers.append(webindex)

webindex_numbers


# In[3]:


# Run the CountVectorizer using the vocabulary
count_vectorizer = CountVectorizer(vocabulary = vocab)

# Fit and transform the preprocessed data to get the BoW representation
count_vectors = count_vectorizer.fit_transform(cleaned_descriptions)


# In[4]:


count_vectors.shape


# ### Word Embeddings Models:
# 
# Choose FastText as the embedding language model.

# In[5]:


# Train the model using words from cleaned descriptions

# Set vector size and corpus file
file = 'cleaned_descriptions.txt'
model = FastText(vector_size = 200)
model.build_vocab(corpus_file = file)

# Train the model
model.train(corpus_file = file, 
                     epochs = model.epochs, 
                     total_examples = model.corpus_count, 
                     total_words = model.corpus_total_words)

# See model overview
print(model)


# In[6]:


# Save the trained FastText model to a file
model_path = 'ft_model.bin'
model.save(model_path)


# In[7]:


# Load trained FastText model
model_path = 'ft_model.bin'
ft_model = FastText.load(model_path)


# In[8]:


# Define a function to generate unweighted word embeddings and also handle missing words
def gen_unweighted(data, model):
    unweighted_word_embeddings = []
    
    for text in data:
        
        # Split text into tokens
        tokens = text.split()
        
        # Initiate an empty list to store unweighted embeddings
        unweighted_embeddings = []

        for token in tokens:
            
            # If the token is in the model's vocabulary, get its embedding
            if token in model.wv.key_to_index:
                word_vec = model.wv.get_vector(token)
                unweighted_embeddings.append(word_vec)
                
            else:
                
                # Handle missing words by replacing with a zero vector
                unweighted_embeddings.append(np.zeros(model.vector_size))

        # Calculate the mean of unweighted word embeddings for this text
        if unweighted_embeddings:
            unweighted_mean_embedding = np.mean(unweighted_embeddings, axis = 0)
            
        else:
            
            # Again, if no valid tokens, use a zero vector
            unweighted_mean_embedding = np.zeros(model.vector_size)

        unweighted_word_embeddings.append(unweighted_mean_embedding)

    return unweighted_word_embeddings


# In[9]:


# Generate unweighted word embeddings with handling missing words for the preprocessed data
unweighted_descriptions = gen_unweighted(cleaned_descriptions, ft_model)

# Print example
unweighted_descriptions[0]


# In[10]:


# Define a function to generate TF-IDF weighted word embeddings
def gen_tfidf_weighted(data, model):
    
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(vocabulary = model.wv.index_to_key)
    tfidf_vectors = tfidf_vectorizer.fit_transform(data)
    
    tfidf_weighted_word_embeddings = []
    
    for tfidf_vector in tfidf_vectors:
        
        # Convert the TF-IDF vector to an array
        tfidf_array = tfidf_vector.toarray()[0]
        
        # Calculate the weighted mean embedding using TF-IDF weights
        weighted_embedding = np.sum(
            tfidf_array[i] * model.wv.get_vector(token) if token in model.wv.key_to_index else np.zeros(model.vector_size)
            for i, token in enumerate(tfidf_vectorizer.get_feature_names_out())
        )
        
        tfidf_weighted_word_embeddings.append(weighted_embedding)
    
    return tfidf_weighted_word_embeddings


# In[11]:


# Generate TF-IDF weighted word embeddings for the preprocessed data
tfidf_descriptions = gen_tfidf_weighted(cleaned_descriptions, ft_model)

# Print example
tfidf_descriptions[0]


# ### Saving outputs
# Save the count vector representations into a file named 
# - count_vector.txt

# In[12]:


# Save the count vectors to a file in the required format
count_vectors_file = "count_vectors.txt"
with open(count_vectors_file, 'w', encoding='utf-8') as f:
    for webindex, count_vector in zip(webindex_numbers, count_vectors):
        # Convert the count vector to a comma-separated string
        count_vector_str = ','.join([f"{i}:{count}" for i, count in enumerate(count_vector.toarray()[0]) if count > 0])
        f.write(f"#{webindex},{count_vector_str}\n")


# ## Task 3. Job Advertisement Classification

# ### Language Models Comparison: Unweighted & TF-IDF Weighted Word Embedding
# 
# Job categories are derived according to each folder name, namely; Accounting_Finance, Engineering, Healthcare_Nursing, and Sales with index number 0, 1, 2, 3 respectively. This categories will be used in the training and testing using various models.

# In[13]:


# Load raw files to retrieve job categories from folder names using integers

job_data = load_files(r"data")  

job_categories = job_data.target
job_categories = [int(c) for c in job_categories]
job_categories


# For Q1, We are using 3 different machine learning models to evaluate the accuracy for each feature representation, including:
# 
# - Logistic Regression Model (Easy to understand and acts as a basic model for comparison)
# - Random Forest Model (Less likely to overfit and can handle noisy features)
# - Support Vector Machine Model (Can handle complicated relationships between features and categories.
# 
# We are using unweighted word embeddings and TF-IDF weighted embeddings as feature representations. Both representations are based on cleaned description from Task 1.

# In[14]:


# Prepare Unweighted and TF-IDF Weighted Descriptions for training and testing

seed = 15
max_iter = 1000

# Split the Unweighted embeddings for descriptions into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_descriptions, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)

# Split the TF-IDF weighted embeddings for descriptions into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(tfidf_descriptions, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)


# In[15]:


# First, choose Logistic Regression Model
lr_model = LogisticRegression(random_state = seed, max_iter = max_iter)


# In[16]:


# Calculate the accuracy score using unweighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
unweighted_description_scores_lr = cross_val_score(lr_model, unweighted_descriptions, 
                                                   job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", unweighted_description_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", unweighted_description_scores_lr.mean())
print("Standard deviation:", unweighted_description_scores_lr.std())


# In[17]:


# Calculate the accuracy score using TF-IDF weighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
tfidf_description_scores_lr = cross_val_score(lr_model, tfidf_descriptions, 
                                              job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of TF-IDF Weighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", tfidf_description_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", tfidf_description_scores_lr.mean())
print("Standard deviation:", tfidf_description_scores_lr.std())


# In[18]:


# Then, try Random Forest Model
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 15)


# In[19]:


# Calculate the accuracy score using unweighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
unweighted_description_scores_rf = cross_val_score(rf_model, unweighted_descriptions,
                                                   job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", unweighted_description_scores_rf)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", unweighted_description_scores_rf.mean())
print("Standard deviation:", unweighted_description_scores_rf.std())


# In[20]:


# Calculate the accuracy score using TF-IDF weighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
tfidf_description_scores_rf = cross_val_score(rf_model, tfidf_descriptions, 
                                              job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", tfidf_description_scores_rf)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", tfidf_description_scores_rf.mean())
print("Standard deviation:", tfidf_description_scores_rf.std())


# In[21]:


# Then, try Support Vector Machine Model
svm_model = SVC(kernel = 'linear', C = 1.0)


# In[22]:


# Calculate the accuracy score using unweighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
unweighted_description_scores_svm = cross_val_score(svm_model, unweighted_descriptions, 
                                                    job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", unweighted_description_scores_svm)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", unweighted_description_scores_svm.mean())
print("Standard deviation:", unweighted_description_scores_svm.std())


# In[23]:


# Calculate the accuracy score using TF-IDF weighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
tfidf_description_scores_svm = cross_val_score(svm_model, tfidf_descriptions, 
                                               job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions:\n")
print("Cross-validation scores:", tfidf_description_scores_svm)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", tfidf_description_scores_svm.mean())
print("Standard deviation:", tfidf_description_scores_svm.std())


# In[24]:


# Set the rounded decimal points
dec = 3

# Print comparison
print("Mean Accuracy\n")
print("Logistic Regression Model")
print("Unweighted:\t\t", round(unweighted_description_scores_lr.mean(), dec))
print("TF-IDF Weighted:\t", round(tfidf_description_scores_lr.mean(), dec))
print("\nRandom Forest Model")
print("Unweighted:\t\t", round(unweighted_description_scores_rf.mean(), dec))
print("TF-IDF Weighted:\t", round(tfidf_description_scores_rf.mean(), dec))
print("\nSupport Vector Machine Model")
print("Unweighted:\t\t", round(unweighted_description_scores_svm.mean(), dec))
print("TF-IDF Weighted:\t", round(tfidf_description_scores_svm.mean(), dec))


# ### Q1 Analysis:
# 
# #### Logistic Regression Model:
# 
# - The TF-IDF weighted feature representation outperforms the unweighted representation significantly.
# - This indicates that TF-IDF weighting helps the logistic regression model better capture the distinguishing features among job advertisements.
# 
# #### Random Forest Model:
# 
# - Though it is slightly lower, the unweighted and TF-IDF weighted representations have similar mean accuracy scores.
# - Random Forest is an ensemble model that may not benefit as much from TF-IDF weighting as logistic regression, which relies on linear relationships.
# 
# #### Support Vector Machine Model
# 
# - Similar to logistic regression, the TF-IDF weighted feature representation performs better.
# - SVM, like logistic regression, benefits from TF-IDF weighting as it helps to create better decision boundaries.
# 
# #### Summary
# 
# In summary, the choice between unweighted and TF-IDF weighted feature representations depends on the specific machine learning model being used. Logistic regression and support vector machines benefit from TF-IDF weighting, as it helps them capture the importance of words. Random Forest, on the other hand, may not show improvement with TF-IDF weighting, as it can handle non-linear relationships differently, but the difference is not very significant. However, looking at this specific model comparison, it could be concluded that the TF-IDF weighted representation consistently performs better across the majority of models in this analysis.

# ### Accuracy Improvement: Descriptions, Titles, and the Combination of Both
# 
# To investigate whether including additional information, such as the title of job advertisements, improves the accuracy of our classification models, we begin with extracting the titles from the raw data. But with a different approach, we will not filter out as much as we did with description, as for the fact that titles contain shorter contents. It is very much similar to Task 1 and Task 2.

# In[25]:


# Extract titles from job data

titles = []

# Define a function to extract the title part from a text
def extract_title(text):
    start_text = text.find("Title: ")
    end_text = text.find("\n")
    if start_text != -1:
        title = text[start_text + len("Title: "):end_text]
        return title
    else:
        return ""

# Iterate through the loaded data and extract titles
for text in job_data.data:
    title = extract_title(text.decode("utf-8"))
    titles.append(title)

# See example of the first title    
emp = 0
titles[emp]


# In[26]:


# Define a function to tokenise data

def tokenize_data(data_raw):
    """
        This function first convert all words to lowercases, 
        it then segment the raw review into sentences and tokenize each sentences 
        and convert the review to a list of tokens.
    """        
    # Convert to lower case
    data_lc = data_raw.lower()
    
    # segament into sentences
    sentences = sent_tokenize(data_lc)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    data_tokenised = list(chain.from_iterable(token_lists))
    return data_tokenised


# In[27]:


# Define a function to print the current stats

def stats_print(data_tk):
    words = list(chain.from_iterable(data_tk))
    vocab = set(words)
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(data_tk))
    lens = [len(article) for article in data_tk]
    print("Average review length:", np.mean(lens))
    print("Maximun review length:", np.max(lens))
    print("Minimun review length:", np.min(lens))
    print("Standard deviation of review length:", np.std(lens))


# In[28]:


# Tokenise the data and compare the result with the original data

title_tk = [tokenize_data(d) for d in titles]

print("Original Data:\n", titles[emp],'\n')
print("Tokenized Data:\n", title_tk[emp])


# In[29]:


# Check overall stats

stats_print(title_tk)


# In[30]:


# Filter out single character tokens
title_tk = [[w for w in words if len(w) >=2] for words in title_tk]

print("Tokenized Data with at least 2 characters:\n", title_tk[emp])


# In[31]:


# Import stop words from the required file

stopwords_file = "stopwords_en.txt"
with open(stopwords_file, 'r') as f:
    stop_words = set(f.read().split())
    
# Filter out stop words

title_tk = [[w for w in words if w not in stop_words] for words in title_tk]

print("Tokenized Data excluding stop words:\n", title_tk[emp])


# In[32]:


# Check stats after cleansing

stats_print(title_tk)


# In[33]:


# Combine tokens and save output as a text file

combined_data = [" ".join(tokens) for tokens in title_tk]
output_file = "cleaned_titles.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    for title in combined_data:
        f.write(title + '\n')


# In[34]:


# Load the cleaned titles
with open("cleaned_titles.txt", "r", encoding="utf-8") as file:
    cleaned_titles = file.readlines()


# We then create feature representations just like what we did for descriptions and then test it with the model. The difference now is that we are not comparing the performance of different models. our main focus is to assess whether the inclusion of additional information, such as the job title, enhances the accuracy of our classification model. We use only the Logistic Regression Model for this comparison by having different combinations of feature representation as followings:
# 
# - Using Only Descriptions
# - Using Only Titles
# - Using Both Description and Titles
# 
# We have done the descriptions in Q1 already, now we have to test for titles and the combination of both. The feature representations for titles will first be created. And later, we will combine it with feature representations for descriptions we have done earlier to come up with the feature representations for the combination of both.

# In[35]:


# Generate unweighted word embeddings with handling missing words for the preprocessed data
unweighted_titles = gen_unweighted(cleaned_titles, ft_model)

# Print example
unweighted_titles[0]


# In[36]:


# Generate TF-IDF weighted word embeddings for the preprocessed data
tfidf_titles = gen_tfidf_weighted(cleaned_titles, ft_model)

# Print example
tfidf_titles[0]


# In[37]:


# Prepare Unweighted and TF-IDF Weighted Titles for training and testing

seed = 15
max_iter = 1000

# Split the unweighted embeddings for titles into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_titles, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)

# Split the TF-IDF weighted embeddings for titles into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(tfidf_titles, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)


# In[38]:


# Calculate the accuracy score using unweighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
unweighted_title_scores_lr = cross_val_score(lr_model, unweighted_titles, 
                                                   job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Titles:\n")
print("Cross-validation scores:", unweighted_title_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", unweighted_title_scores_lr.mean())
print("Standard deviation:", unweighted_title_scores_lr.std())


# In[39]:


# Calculate the accuracy score using TF-IDF weighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
tfidf_title_scores_lr = cross_val_score(lr_model, tfidf_titles, 
                                              job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of TF-IDF Weighted Word Embeddings for Titles:\n")
print("Cross-validation scores:", tfidf_title_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", tfidf_title_scores_lr.mean())
print("Standard deviation:", tfidf_title_scores_lr.std())


# And now we are combining the two representations then test them for the last set of comparison.

# In[40]:


# Concatenate unweighted word embeddings for titles and descriptions horizontally
unweighted_combined = np.hstack((unweighted_descriptions, unweighted_titles))
tfidf_combined = np.hstack((tfidf_descriptions, tfidf_titles))


# In[41]:


# Prepare Unweighted and TF-IDF Weighted Descriptions and Titles for training and testing

seed = 15
max_iter = 1000

# Split the unweighted embeddings for both descriptions and titles into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(unweighted_combined, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)

# Split the TF-IDF weighted embeddings for both descriptions and titles into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(tfidf_combined, 
                                                                                 job_categories, 
                                                                                 list(range(0,len(job_categories))),
                                                                                 test_size = 0.2, 
                                                                                 random_state = seed)


# In[42]:


# Calculate the accuracy score using unweighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
unweighted_combined_scores_lr = cross_val_score(lr_model, unweighted_combined, 
                                                job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of Unweighted Word Embeddings for Descriptions and Titles:\n")
print("Cross-validation scores:", unweighted_combined_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", unweighted_combined_scores_lr.mean())
print("Standard deviation:", unweighted_combined_scores_lr.std())


# In[43]:


# Calculate the accuracy score using TF-IDF weighted word embeddings

# Perform 5-fold cross-validation and specify the scoring metric
tfidf_combined_scores_lr = cross_val_score(lr_model, tfidf_combined, 
                                           job_categories, cv = 5, scoring = 'accuracy')

# Print the cross-validation scores
print("Results of TF-IDF Weighted Word Embeddings for Descriptions and Titles:\n")
print("Cross-validation scores:", tfidf_combined_scores_lr)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", tfidf_combined_scores_lr.mean())
print("Standard deviation:", tfidf_combined_scores_lr.std())


# In[44]:


# Set the rounded decimal points
dec = 3

# Print comparison
print("Mean Accuracy\n")
print("Unweighted")
print("Only Descriptions:\t", round(unweighted_description_scores_lr.mean(), dec))
print("Only Titles:\t\t", round(unweighted_title_scores_lr.mean(), dec))
print("Combination:\t\t", round(unweighted_combined_scores_lr.mean(), dec))
print("\nTF-IDF Weighted")
print("Only Descriptions:\t", round(tfidf_description_scores_lr.mean(), dec))
print("Only Titles:\t\t", round(tfidf_title_scores_lr.mean(), dec))
print("Combination:\t\t", round(tfidf_combined_scores_lr.mean(), dec))


# ### Q2 Analysis:
# 
# #### Unweighted Representations:
# 
# - Looking at the mean accuracy achieved, it suggests that using only job descriptions alone for classification leads to relatively low accuracy.
# - When using only job titles, the mean accuracy increases. This indicates that titles contribute additional information and improve classification performance compared to descriptions alone.
# - By combining both job descriptions and titles, the mean accuracy further improves. This suggests that leveraging both sources of information results in better classification accuracy than using either one individually.
# 
# #### TF-IDF Weighted Representations:
# 
# - When considering only descriptions, the mean accuracy significantly improves compared to the unweighted representation. TF-IDF weighting seems to enhance the model's ability to classify job advertisements based on descriptions.
# - When using only job titles, the mean accuracy is also getting better compared to unweighted representations. This indicates that job titles provide valuable information, and TF-IDF weighting seems to have as substantial an impact here as it does with descriptions.
# - Combining both job descriptions and titles while using TF-IDF weighting results in the best mean accuracy score. This demonstrates that the combination of both textual sources, when weighted with TF-IDF, yields a significant impact on classification accuracy.
# 
# #### Summary
# 
# TF-IDF weighting generally improves classification accuracy compared to unweighted representations for both descriptions and titles. It is also important to point out that combining job descriptions and titles consistently leads to improved accuracy across both unweighted and TF-IDF weighted scenarios. TF-IDF weighted descriptions alone also achieve  quite high accuracy score, emphasising their significance in the classification process. Still, in summary, the results indicate that including both job descriptions and titles, especially when TF-IDF Weighted is applied, leads to the best classification performance.

# ## Summary
# 
# These tasks helped preprocess and represent job advertisement data effectively and evaluate different models for categorising job advertisements. The choice of models, feature representations, and data combination strategies impacted classification accuracy. To improve job advertisement classification further, we can consider exploring and analysing the data, fine-tuning model settings, incorporating additional features, experimenting with various machine learning models, and utilising more advanced natural language processing tools. Though I'm not an expert, through this assignment, I'm getting a better picture of Data Science.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>
