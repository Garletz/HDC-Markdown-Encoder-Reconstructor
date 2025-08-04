import urllib.request
from pathlib import Path

def download_word_list(url, output_file):
    print(f"Downloading vocabulary from {url} ...")
    urllib.request.urlretrieve(url, output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Source: 50k most common English words (Google 10000/50k/100k)
    # https://github.com/first20hours/google-10000-english/blob/master/20k.txt
    # We'll use a direct raw link for automation
    vocab_url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
    output = "english_vocab.txt"
    download_word_list(vocab_url, output)
