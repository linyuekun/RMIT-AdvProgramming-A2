#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
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
# 
# The data are separated in 4 sub-folders, which can also be identified as 4 job categories, containing the total of 776 job files. The primary goal for this task is to prepare the raw data for subsequent analysis and model building. This foundational step involves mainly on data cleaning and text preprocessing. We will focus on understanding the data's characteristics, handling missing values, and transforming text data into a suitable format for natural language processing (NLP) tasks.

# ## Importing libraries 

# In[1]:


from sklearn.datasets import load_files  
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from collections import Counter
import numpy as np
import os


# ### 1.1 Examining and loading data
# - Examine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# In[2]:


# Load job data
job_data = load_files(r"data")


# In[3]:


# Extract descriptions from job data
descriptions = []

# Define a function to extract the description part from a text
def extract_description(text):
    start_text = text.find("Description: ")
    if start_text != -1:
        description = text[start_text + len("Description: "):]
        return description
    else:
        return ""

# Iterate through the loaded data and extract descriptions
for text in job_data.data:
    description = extract_description(text.decode("utf-8"))  # Decode bytes to string
    descriptions.append(description)

# See example of the first description    
emp = 0
descriptions[emp]


# In[4]:


# Extract webindex from job data

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


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# We begin with defining functions for tokenising data and printing stats. Within the tokenising function, we use <span style="color: red"> r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" </span> as a regular expression. We also transform every word into lower-case.

# In[5]:


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


# In[6]:


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


# In[7]:


# Tokenise the data and compare the result with the original data

data_tk = [tokenize_data(d) for d in descriptions]

print("Original Data:\n",descriptions[emp],'\n')
print("Tokenized Data:\n",data_tk[emp])


# In[8]:


# First check point for overall stats

stats_print(data_tk)


# After tokenising the data, we can now filter out required pre-processing steps:
# 
# - Remove words with length less than 2
# - Remove stopwords using the provided stop words list (i.e, stopwords_en.txt)
# - Remove the words that appear only once in the document collection, based on term frequency
# - Remove the top 50 most frequent words based on document frequency

# In[9]:


# Check all the single character tokens

# Create a list of single character token for each review
st_list = [[w for w in words if len(w) < 2 ] for words in data_tk] 

# Merge them together in one list
list(chain.from_iterable(st_list))


# In[10]:


# Filter out single character tokens
data_tk = [[w for w in words if len(w) >=2] for words in data_tk]

print("Tokenized Data with at least 2 characters:\n", data_tk[emp])


# In[11]:


# Check stats after eliminating single character tokens

stats_print(data_tk)


# In[12]:


# Import stop words from the required file

stopwords_file = "stopwords_en.txt"
with open(stopwords_file, 'r') as f:
    stop_words = set(f.read().split())
stop_words


# In[13]:


# Filter out stop words

data_tk = [[w for w in words if w not in stop_words] for words in data_tk]

print("Tokenized Data excluding stop words:\n", data_tk[emp])


# In[14]:


# Check stats after eliminating stop words

stats_print(data_tk)


# In[15]:


# Check counts for each remaining words

word_counts = Counter(w for words in data_tk for w in words)
word_counts


# In[16]:


# Filter out words that appear only once

data_tk = [[w for w in words if word_counts[w] > 1] for words in data_tk]

print("Tokenized Data with more than one occurance:\n", data_tk[emp])


# In[17]:


# Check stats after eliminating words that appear only once

stats_print(data_tk)


# In[18]:


# Check 50 most common words

# Indicate _ to only include words in the list
most_common_words = [w for w, _ in word_counts.most_common(50)]

# Check with the numbers
most_common_words_count = [w for w in word_counts.most_common(50)]
most_common_words_count


# In[19]:


# Filter out top 50 most common words based on document frequency

data_tk = [[w for w in words if w not in most_common_words] for words in data_tk]

print("Tokenized Data without 50 most frequent words:\n", data_tk[emp])


# In[20]:


# Check stats after eliminating 50 most common words 

stats_print(data_tk)


# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[21]:


# Combine tokens and save output as a text file

combined_data = [" ".join(tokens) for tokens in data_tk]
output_file = "cleaned_descriptions.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    for description in combined_data:
        f.write(description + '\n')


# In[22]:


# Save a list of sorted vocabs as a text file

unique_words = set(w for words in data_tk for w in words)
sorted_unique_words = sorted(unique_words)

vocab_file = "vocab.txt"

with open(vocab_file, 'w') as f:
    for index, word in enumerate(sorted_unique_words):
        f.write(f"{word}:{index}\n")


# ## Summary
# 
# By completing this task, we have set the stage for more advanced analyses and modeling in subsequent tasks. Our clean and well-structured dataset, along with the insights gained through exploratory data analysis, will enable us to build robust machine learning models and extract meaningful information from the data. The outputs from this task will be used further in the following tasks.

# ## Couple of notes for all code blocks in this notebook
# - please provide proper comment on your code
# - Please re-start and run all cells to make sure codes are runable and include your output in the submission.   
# <span style="color: red"> This markdown block can be removed once the task is completed. </span>
