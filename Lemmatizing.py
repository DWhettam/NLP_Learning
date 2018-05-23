from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos="a")) # = Good
print(lemmatizer.lemmatize("better")) # = better
