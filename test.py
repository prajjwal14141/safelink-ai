import pandas as pd

# Load the dataset
dataset_path = './data/data.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows
print(data.head())

# Check labels for known good URLs
known_good_urls = ['wikipedia.com', 'google.com', 'facebook.com', 'twitter.com', 'github.com']

for url in known_good_urls:
    if url in data['url'].values:
        print(f"{url}: {data[data['url'] == url]['label'].values[0]}")
    else:
        print(f"{url}: Not found in dataset")
