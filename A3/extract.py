import pandas as pd

# Read the processed RT data
df = pd.read_csv('processed_RTs.tsv', sep='\t')

# Read the GPT-3 data
gpt3_df = pd.read_csv('all_stories_gpt3.csv')

# Group by word to get unique words with their statistics
word_stats = df.groupby('word').agg({
    'meanItemRT': 'first',
    'nItem': 'first'
}).reset_index()

# Remove special characters from words and convert to lowercase
word_stats['word'] = word_stats['word'].str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()

# Add word length column (after removing special characters)
word_stats['word_length'] = word_stats['word'].str.len()

# Create normalized versions
word_stats['word_normalized'] = word_stats['word']
gpt3_df['token_normalized'] = gpt3_df['token'].str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()

# Calculate average log probability for each normalized token from GPT-3
gpt3_probs = gpt3_df.groupby('token_normalized')['logprob'].mean().reset_index()
gpt3_probs.columns = ['word_normalized', 'gpt3_logprob']

# Merge with word stats using normalized versions
word_stats = word_stats.merge(gpt3_probs, on='word_normalized', how='left')

# Negate the gpt3_logprob column
word_stats['gpt3_logprob'] = -word_stats['gpt3_logprob']

# Define function words
function_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which',
    'who', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
    'must', 'shall', 'ought', 'not', 'no', 'nor', 'so', 'than'
}

# Classify words as content or function
word_stats['word_type'] = word_stats['word'].apply(
    lambda x: 'function' if x in function_words else 'content'
)

# Drop the normalized column, keep cleaned word
word_stats = word_stats[['word', 'meanItemRT', 'nItem', 'word_length', 'gpt3_logprob', 'word_type']]
word_stats = word_stats.groupby('word').agg({
    'meanItemRT': 'mean',
    'nItem': 'sum',
    'word_length': 'first',
    'gpt3_logprob': 'mean',
    'word_type': 'first'
}).reset_index()

# Save to CSV
word_stats.to_csv('word_statistics.csv', index=False)

print(f"Created word_statistics.csv with {len(word_stats)} unique words")
print(f"Words with GPT-3 probabilities: {word_stats['gpt3_logprob'].notna().sum()}")
print(f"Content words: {(word_stats['word_type'] == 'content').sum()}")
print(f"Function words: {(word_stats['word_type'] == 'function').sum()}")