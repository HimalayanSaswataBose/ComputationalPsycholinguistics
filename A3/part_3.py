import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

class FOBSModel:
    """
    Frequency Ordered Bin Search (FOBS) model for human memory.
    Models word recognition by organizing words into frequency-ordered bins.
    """
    
    def __init__(self, num_bins: int = 10):
        """
        Initialize the FOBS model.
        
        Args:
            num_bins: Number of frequency bins to organize words into
        """
        self.num_bins = num_bins
        self.word_data = None
        self.lemma_bins = {}
        self.surface_form_bins = {}
        self.lemma_freq = defaultdict(int)
        self.surface_freq = defaultdict(int)
        
    def load_data(self, csv_path: str) -> None:
        """Load word statistics from CSV file."""
        self.word_data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.word_data)} word entries")
        
    def lemmatize(self, word: str) -> str:
        """
        Lemmatize a word using NLTK's WordNetLemmatizer.
        
        Args:
            word: Word to lemmatize
            
        Returns:
            Lemmatized form of the word
        """
        lemmatizer = WordNetLemmatizer()
        word = word.lower()
        
        # Try lemmatizing as different parts of speech and pick the shortest
        lemma_noun = lemmatizer.lemmatize(word, pos='n')
        lemma_verb = lemmatizer.lemmatize(word, pos='v')
        lemma_adj = lemmatizer.lemmatize(word, pos='a')
        
        # Return the shortest lemma (most reduced form)
        return min([lemma_noun, lemma_verb, lemma_adj], key=len)
    
    def calculate_frequencies(self) -> None:
        """Calculate frequency of lemmas and surface forms from data."""
        for _, row in self.word_data.iterrows():
            word = row['word']
            n_items = row['nItem']
            
            # Calculate frequencies (occurrences)
            self.surface_freq[word] += 1
            lemma = self.lemmatize(word)
            self.lemma_freq[lemma] += 1
            
        print(f"Found {len(self.lemma_freq)} unique lemmas")
        print(f"Found {len(self.surface_freq)} unique surface forms")
    
    def create_bins(self) -> None:
        """Organize lemmas and surface forms into frequency-ordered bins."""
        # Sort by frequency (descending)
        sorted_lemmas = sorted(self.lemma_freq.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        sorted_surfaces = sorted(self.surface_freq.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        # Distribute into bins
        lemmas_per_bin = len(sorted_lemmas) // self.num_bins
        surfaces_per_bin = len(sorted_surfaces) // self.num_bins
        
        for i in range(self.num_bins):
            start_idx = i * lemmas_per_bin
            end_idx = start_idx + lemmas_per_bin if i < self.num_bins - 1 else len(sorted_lemmas)
            self.lemma_bins[i] = sorted_lemmas[start_idx:end_idx]
            
            start_idx = i * surfaces_per_bin
            end_idx = start_idx + surfaces_per_bin if i < self.num_bins - 1 else len(sorted_surfaces)
            self.surface_form_bins[i] = sorted_surfaces[start_idx:end_idx]
        
        print(f"Created {self.num_bins} bins for lemmas and surface forms")
    
    def get_bin_index(self, word: str, use_lemma: bool = True) -> int:
        """
        Find which bin a word belongs to.
        
        Args:
            word: Word to search for
            use_lemma: Whether to use lemmatized form
            
        Returns:
            Bin index (0 to num_bins-1) or -1 if not found
        """
        search_word = self.lemmatize(word) if use_lemma else word
        bins = self.lemma_bins if use_lemma else self.surface_form_bins
        
        for bin_idx, bin_words in bins.items():
            if any(w[0] == search_word for w in bin_words):
                return bin_idx
        
        return -1
    
    def predict_rt(self, word: str, base_rt: float = 300.0, 
                   bin_cost: float = 50.0) -> float:
        """
        Predict reaction time based on FOBS model.
        
        Args:
            word: Word to predict RT for
            base_rt: Base reaction time in ms
            bin_cost: Additional cost per bin search in ms
            
        Returns:
            Predicted reaction time in ms
        """
        bin_idx = self.get_bin_index(word, use_lemma=True)
        
        if bin_idx == -1:
            # Word not found, assume worst case
            return base_rt + (self.num_bins * bin_cost)
        
        # RT increases with bin index (lower frequency = higher bin number)
        return base_rt + (bin_idx * bin_cost)
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model predictions against actual RT data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        actuals = []
        
        for _, row in self.word_data.iterrows():
            word = row['word']
            actual_rt = row['meanItemRT']
            
            predicted_rt = self.predict_rt(word)
            predictions.append(predicted_rt)
            actuals.append(actual_rt)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'n_predictions': len(predictions)
        }
        
        return metrics
    
    def get_bin_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each bin.
        
        Returns:
            DataFrame with bin statistics
        """
        stats = []
        
        for bin_idx in range(self.num_bins):
            lemma_words = self.lemma_bins[bin_idx]
            
            bin_stat = {
                'bin_index': bin_idx,
                'n_lemmas': len(lemma_words),
                'avg_frequency': np.mean([freq for _, freq in lemma_words]),
                'min_frequency': min([freq for _, freq in lemma_words]),
                'max_frequency': max([freq for _, freq in lemma_words])
            }
            stats.append(bin_stat)
        
        return pd.DataFrame(stats)
    
    def search_word(self, word: str) -> Dict:
        """
        Search for a word and return detailed information.
        
        Args:
            word: Word to search for
            
        Returns:
            Dictionary with word information
        """
        lemma = self.lemmatize(word)
        bin_idx = self.get_bin_index(word, use_lemma=True)
        
        # Get actual RT if available
        actual_rt = None
        word_data = self.word_data[self.word_data['word'] == word]
        if not word_data.empty:
            actual_rt = word_data['meanItemRT'].mean()
        
        return {
            'surface_form': word,
            'lemma': lemma,
            'bin_index': bin_idx,
            'lemma_frequency': self.lemma_freq[lemma],
            'surface_frequency': self.surface_freq[word],
            'predicted_rt': self.predict_rt(word),
            'actual_rt': actual_rt
        }


def main():
    """Main function to demonstrate FOBS model and test hypothesis."""
    
    # Initialize model
    print("=" * 60)
    print("FOBS Model: Frequency Ordered Bin Search")
    print("=" * 60)
    
    model = FOBSModel(num_bins=10)
    
    # Load data
    print("\n1. Loading data...")
    model.load_data('word_statistics.csv')
    
    # Calculate frequencies
    print("\n2. Calculating frequencies...")
    model.calculate_frequencies()
    
    # Create bins
    print("\n3. Creating frequency-ordered bins...")
    model.create_bins()
    
    # Display bin statistics
    print("\n4. Bin Statistics:")
    bin_stats = model.get_bin_statistics()
    print(bin_stats.to_string(index=False))
    
    # Evaluate model
    print("\n5. Model Evaluation:")
    metrics = model.evaluate_model()
    print(f"  Mean Absolute Error: {metrics['mae']:.2f} ms")
    print(f"  Root Mean Squared Error: {metrics['rmse']:.2f} ms")
    print(f"  Correlation: {metrics['correlation']:.3f}")
    print(f"  Number of predictions: {metrics['n_predictions']}")
    
    # Hypothesis Testing: Surface vs Lemma Frequency
    print("\n6. Hypothesis Testing: Surface vs Lemma Frequency")
    print("=" * 60)
    
    # Prepare data for regression models
    word_data_list = []
    for _, row in model.word_data.iterrows():
        word = row['word']
        lemma = model.lemmatize(word)
        
        word_data_list.append({
            'word': word,
            'rt': row['meanItemRT'],
            'surface_freq': model.surface_freq[word],
            'surface_length': len(word),
            'lemma': lemma,
            'lemma_freq': model.lemma_freq[lemma],
            'lemma_length': len(lemma)
        })
    
    df = pd.DataFrame(word_data_list)
    
    # Model 1: RT ~ surface_freq + surface_length
    print("\nModel 1: RT ~ word_freq + word_length")
    X1 = df[['surface_freq', 'surface_length']].values
    y = df['rt'].values
    
    reg1 = LinearRegression()
    reg1.fit(X1, y)
    y_pred1 = reg1.predict(X1)
    
    r2_1 = r2_score(y, y_pred1)
    mae_1 = np.mean(np.abs(y - y_pred1))
    rmse_1 = np.sqrt(np.mean((y - y_pred1) ** 2))
    
    print(f"  R² Score: {r2_1:.4f}")
    print(f"  MAE: {mae_1:.2f} ms")
    print(f"  RMSE: {rmse_1:.2f} ms")
    print(f"  Coefficients: freq={reg1.coef_[0]:.4f}, length={reg1.coef_[1]:.4f}")
    print(f"  Intercept: {reg1.intercept_:.2f}")
    
    # Model 2: RT ~ lemma_freq + lemma_length
    print("\nModel 2: RT ~ lemma_freq + lemma_length")
    X2 = df[['lemma_freq', 'lemma_length']].values
    
    reg2 = LinearRegression()
    reg2.fit(X2, y)
    y_pred2 = reg2.predict(X2)
    
    r2_2 = r2_score(y, y_pred2)
    mae_2 = np.mean(np.abs(y - y_pred2))
    rmse_2 = np.sqrt(np.mean((y - y_pred2) ** 2))
    
    print(f"  R² Score: {r2_2:.4f}")
    print(f"  MAE: {mae_2:.2f} ms")
    print(f"  RMSE: {rmse_2:.2f} ms")
    print(f"  Coefficients: freq={reg2.coef_[0]:.4f}, length={reg2.coef_[1]:.4f}")
    print(f"  Intercept: {reg2.intercept_:.2f}")
    
    # Compare models
    print("\nModel Comparison:")
    print(f"  R² Improvement (Model 2 vs Model 1): {r2_2 - r2_1:.4f}")
    print(f"  MAE Improvement: {mae_1 - mae_2:.2f} ms")
    print(f"  RMSE Improvement: {rmse_1 - rmse_2:.2f} ms")
    
    if r2_2 > r2_1:
        print("\n  ✓ Hypothesis SUPPORTED: Lemma frequency predicts RT better than surface frequency")
    else:
        print("\n  ✗ Hypothesis NOT SUPPORTED: Surface frequency predicts RT better than lemma frequency")
    
    # Hypothesis Testing: Pseudo-affixed vs Real-affixed Words
    print("\n8. Hypothesis Testing: Pseudo-affixed vs Real-affixed Words")
    print("=" * 60)
    print("Hypothesis: Pseudo-affixed words (e.g., 'finger') take more processing")
    print("            time than words with real affixes (e.g., 'golden')")
    print()
    
    # Define test words with pseudo-affixes (suffix-like but not morphologically decomposable)
    pseudo_affixed = [
        'finger',    # '-er' is not a suffix (not 'fing' + 'er')
        'corner',    # '-er' is not a suffix (not 'corn' + 'er')
        'based',    # '-ed' is not a suffix (not 'bas' + 'ed')
        'admiral',    # '-al' is not a suffix (not 'admir' + 'al')
        'number'     # '-er' is not a suffix (not 'numb' + 'er')
    ]
    
    # Define words with real affixes (morphologically decomposable)
    real_affixed = [
        'golden',    # gold + -en (adjectival suffix)
        'higher',    # high + -er (comparative suffix)
        'flying',    # fly + -ing (gerund suffix)
        'seeing',    # see + -ing (gerund suffix)
        'burning'     # burn + -ing (gerund suffix)
    ]
    
    print("Test Words:")
    print(f"  Pseudo-affixed: {', '.join(pseudo_affixed)}")
    print(f"  Real-affixed: {', '.join(real_affixed)}")
    print()
    
    # Collect data for both groups
    pseudo_data = []
    real_data = []
    
    print("Pseudo-affixed Words Analysis:")
    print("-" * 60)
    for word in pseudo_affixed:
        word_rows = model.word_data[model.word_data['word'].str.lower() == word.lower()]
        if not word_rows.empty:
            rt = word_rows['meanItemRT'].mean()
            lemma = model.lemmatize(word)
            freq = model.lemma_freq[lemma]
            length = len(word)
            
            pseudo_data.append({
                'word': word,
                'rt': rt,
                'length': length,
                'frequency': freq,
                'lemma': lemma
            })
            
            print(f"  {word:12s} | RT: {rt:6.2f} ms | Length: {length} | Freq: {freq:3d} | Lemma: {lemma}")
        else:
            print(f"  {word:12s} | NOT FOUND IN DATA")
        
    print()
    print("Real-affixed Words Analysis:")
    print("-" * 60)
    for word in real_affixed:
        word_rows = model.word_data[model.word_data['word'].str.lower() == word.lower()]
        if not word_rows.empty:
            rt = word_rows['meanItemRT'].mean()
            lemma = model.lemmatize(word)
            freq = model.lemma_freq[lemma]
            length = len(word)
            
            real_data.append({
                'word': word,
                'rt': rt,
                'length': length,
                'frequency': freq,
                'lemma': lemma
            })
            
            print(f"  {word:12s} | RT: {rt:6.2f} ms | Length: {length} | Freq: {freq:3d} | Lemma: {lemma}")
        else:
            print(f"  {word:12s} | NOT FOUND IN DATA")
    
    # Statistical comparison
    if len(pseudo_data) > 0 and len(real_data) > 0:
        pseudo_rts = [d['rt'] for d in pseudo_data]
        real_rts = [d['rt'] for d in real_data]
        
        pseudo_mean = np.mean(pseudo_rts)
        real_mean = np.mean(real_rts)
        
        pseudo_std = np.std(pseudo_rts)
        real_std = np.std(real_rts)
        
        print()
        print("Statistical Summary:")
        print("-" * 60)
        print(f"  Pseudo-affixed words:")
        print(f"    N = {len(pseudo_data)}")
        print(f"    Mean RT = {pseudo_mean:.2f} ms (SD = {pseudo_std:.2f})")
        print(f"    Avg Length = {np.mean([d['length'] for d in pseudo_data]):.1f}")
        print(f"    Avg Frequency = {np.mean([d['frequency'] for d in pseudo_data]):.1f}")
        
        print()
        print(f"  Real-affixed words:")
        print(f"    N = {len(real_data)}")
        print(f"    Mean RT = {real_mean:.2f} ms (SD = {real_std:.2f})")
        print(f"    Avg Length = {np.mean([d['length'] for d in real_data]):.1f}")
        print(f"    Avg Frequency = {np.mean([d['frequency'] for d in real_data]):.1f}")
        
        print()
        print(f"  Difference in Mean RT: {pseudo_mean - real_mean:.2f} ms")
        print(f"  Effect Size (Cohen's d): {(pseudo_mean - real_mean) / np.sqrt((pseudo_std**2 + real_std**2) / 2):.3f}")
        
        # Perform t-test if we have enough data
        if len(pseudo_rts) >= 3 and len(real_rts) >= 3:
            t_stat, p_value = stats.ttest_ind(pseudo_rts, real_rts)
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
        
        print()
        print("Interpretation:")
        print("-" * 60)
        if pseudo_mean > real_mean:
            diff_pct = ((pseudo_mean - real_mean) / real_mean) * 100
            print(f"  ✓ Hypothesis SUPPORTED: Pseudo-affixed words show {diff_pct:.1f}% longer")
            print(f"    processing time ({pseudo_mean:.2f} ms vs {real_mean:.2f} ms).")
            print()
        else:
            diff_pct = ((real_mean - pseudo_mean) / pseudo_mean) * 100
            print(f"  ✗ Hypothesis NOT SUPPORTED: Real-affixed words show {diff_pct:.1f}% longer")
            print(f"    processing time ({real_mean:.2f} ms vs {pseudo_mean:.2f} ms).")
            print()
    else:
        print()
        print("  ⚠ Insufficient data to test hypothesis (words not found in dataset)")
    
    # # Example word searches
    # print("\n7. Example Word Searches:")
    # example_words = ['the', 'beautiful', 'extraordinary', 'bubble']
    
    # for word in example_words:
    #     info = model.search_word(word)
    #     print(f"\n  Word: '{word}'")
    #     print(f"    Lemma: {info['lemma']}")
    #     print(f"    Bin: {info['bin_index']}")
    #     print(f"    Lemma Frequency: {info['lemma_frequency']}")
    #     print(f"    Predicted RT: {info['predicted_rt']:.2f} ms")
    #     if info['actual_rt']:
    #         print(f"    Actual RT: {info['actual_rt']:.2f} ms")
    
    print("\n" + "=" * 60)
    print("FOBS Model Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Download required NLTK data (only needed once)
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    main()