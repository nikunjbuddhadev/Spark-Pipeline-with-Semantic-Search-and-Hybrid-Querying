# Configuration for the Pipeline

# Currency conversion rates to USD
CURRENCY_RATES = {
    'EUR': 1.1,
    'GBP': 1.27,
    'JPY': 1/150
}

# Category standardization mapping
CATEGORY_MAPPING = {
    'ai': 'AI_ML',
    'ml': 'AI_ML',
    'machine learning': 'AI_ML',
    'artificial intelligence': 'AI_ML',
    'ai & ml': 'AI_ML',
    'ai/ml': 'AI_ML',
    'fintech': 'FinTech',
    'financial technology': 'FinTech',
    'cloud computing': 'Cloud',
    'security': 'Security',
    'data analytics': 'Data Analytics',
    'analytics': 'Data Analytics',
    'software': 'Software',
    'saas': 'SaaS',
}

# Acronym / alias map: normalized alias -> canonical company (metadata name).
# Only for abbreviations that string similarity cannot resolve on its own.
COMPANY_ALIASES = {
    'aws': 'Amazon Web Services',
    'azure': 'Microsoft',
    'msft': 'Microsoft',
}

# Minimum fuzzy score (0-100) to accept a company match
COMPANY_MATCH_THRESHOLD = 80

# Company size thresholds
COMPANY_SIZE_THRESHOLDS = {
    'Small': (0, 10000),
    'Medium': (10000, 30000),
    'Large': (30000, float('inf'))
}

# Filtering criteria for AI articles export
AI_ARTICLES_FILTERS = {
    'start_date': '2022-01-01',
    'end_date': '2024-12-31',
    'min_arr_usd': 50_000_000
}

# Output paths
OUTPUT_DIR = './data_outputs'
