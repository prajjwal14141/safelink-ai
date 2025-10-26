import re
import math
from collections import Counter

# --- Define High-Risk Tokens Globally ---
HIGH_RISK_TOKENS = [
    'exe', 'php', 'install', 'toolbar', 'crack', 'spider', 'lucky',
    'admin', 'login', 'secure', 'account', 'password', 'key', 'download',
    'free', 'gift', 'prize', 'winner', 'click'
]

def clean_url(url):
    """ Cleans URL: removes protocol, www., trailing slash """
    try:
        url = str(url)
        url = re.sub(r'^(https?|ftp)://', '', url)
        url = re.sub(r'^www\.', '', url)
        url = re.sub(r'/$', '', url)
        return url
    except Exception as e:
        print(f"Error cleaning URL {url}: {e}")
        return ""

def getTokens(input):
    """ Custom tokenizer: splits by '/', '-', '.' """
    try:
        tokensBySlash = str(input).split('/')
        allTokens = []
        for i in tokensBySlash:
            tokens = str(i).split('-')
            tokensByDot = []
            for j in range(0, len(tokens)):
                tempTokens = str(tokens[j]).split('.')
                tokensByDot = tokensByDot + tempTokens
            allTokens = allTokens + tokens + tokensByDot
        allTokens = list(set(allTokens))
        # Remove common/unhelpful tokens, ensure lowercase, ignore short tokens
        common_tokens = ['com', 'www', 'http', 'https', 'org', 'net', 'io', 'co', 'uk', 'html', 'htm']
        allTokens = [t.lower() for t in allTokens if t.lower() not in common_tokens and len(t) > 1]
        return allTokens
    except Exception as e:
        print(f"Error tokenizing input {input}: {e}")
        return []

def entropy(s):
    """ Calculates Shannon entropy """
    p, lns = Counter(s), float(len(s))
    if lns == 0: return 0.0
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())
