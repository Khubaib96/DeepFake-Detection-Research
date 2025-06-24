"""
Advanced Audio Deepfake Detection with Parameter Optimization
============================================================

This script implements comprehensive parameter optimization for deepfake detection
using multiple spectral features, visual discriminability metrics, and statistical analysis.

Key Features:
1. Grid search across all parameter combinations
2. Visual discriminability scoring using computer vision metrics
3. Multiple spectrogram generation methods
4. Statistical discrimination analysis
5. Multi-objective optimization with weighted scoring

Author: Advanced Audio Analysis for Deepfake Detection
Date: 2024
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fftpack import dct
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import os
import seaborn as sns
from pathlib import Path
import warnings
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import json
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from sklearn.metrics import mutual_info_score
import time
from tqdm import tqdm
import pickle
import random
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

class OptimalDeepfakeAnalyzer:
    """
    Advanced analyzer for finding optimal parameters for deepfake detection
    using comprehensive grid search with visual and statistical metrics.
    """
    
    def __init__(self):
        """Initialize the optimizer with parameter grids and scoring weights."""
        
        # Define strategic parameter search space for comprehensive evaluation
        # -------------------------------------------------------------
        # Reduced but representative parameter grid that ensures good coverage
        # across all dimensions while keeping total combinations manageable.
        # This allows testing all parameter variations within reasonable time.
        #   • sample_rate   – Key rates: telephony, standard, high-quality
        #   • n_fft         – Representative FFT sizes covering the range
        #   • hop_length_ratio – Different overlap strategies
        #   • window        – Most common analysis windows
        #   • n_lfcc / n_mfcc – Representative coefficient counts
        #   • n_mels        – Key mel filter-bank resolutions
        # -------------------------------------------------------------
        self.parameter_grid = {
            'sample_rate': [8000, 16000, 44100, 48000],    # 4 values - full range
            'n_fft': [128, 256, 512, 1024, 2048],               # 5 values - comprehensive range
            'hop_length_ratio': [ 0.25, 0.5, 0.75],   # 4 values - full overlap range
            'window': ['hann', 'hamming', 'blackman'],      # 3 values - common windows
            'n_lfcc': [20, 40, 60, 80, 100],                # 5 values - full coefficient range
            'n_mfcc': [13, 20, 26, 39],                     # 4 values - standard range
            'n_mels': [40, 64, 80, 128, 256]                # 5 values - comprehensive mel range
        }
        
        # Scoring weights for multi-objective optimization
        self.scoring_weights = {
            'statistical_discrimination': 0.15,  # Increased weight
            'visual_discriminability': 0.70,     # Maintained focus on visual plotting
            'pattern_consistency': 0.15          # Maintained consistency weight
        }
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
        print("🔍 Optimal Deepfake Analyzer Initialized")
        print(f"📊 Parameter combinations to test: {self._count_combinations()}")
        print(f"⚖️  Scoring weights: {self.scoring_weights}")
        
        # Show first few combinations to verify parameter variation
        print("\n🔍 First 10 parameter combinations to verify variation:")
        combinations = list(self._generate_parameter_combinations())
        for i, combo in enumerate(combinations[:10]):
            print(f"  {i+1:2d}. SR:{combo['sample_rate']:5d} FFT:{combo['n_fft']:4d} Hop:{combo['hop_length']:3d} "
                  f"Win:{combo['window']:8s} LFCC:{combo['n_lfcc']:2d} MFCC:{combo['n_mfcc']:2d} Mels:{combo['n_mels']:3d}")
        
        print("-" * 60)
    
    def _count_combinations(self):
        """Count total parameter combinations."""
        count = 1
        for key, values in self.parameter_grid.items():
            count *= len(values)
        return count
    
    def _generate_parameter_combinations(self):
        """Generate all parameter combinations for grid search."""
        keys = self.parameter_grid.keys()
        values = self.parameter_grid.values()
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            # Calculate hop_length from ratio
            params['hop_length'] = int(params['n_fft'] * params['hop_length_ratio'])
            del params['hop_length_ratio']  # Remove ratio, keep actual hop_length
            yield params
    
    def _generate_smart_parameter_samples(self, n_samples=50, exploration_factor=0.3):
        """
        Generate smart parameter samples using adaptive exploration strategy.
        
        Parameters:
        -----------
        n_samples : int
            Number of parameter combinations to sample
        exploration_factor : float
            Balance between exploitation of good regions and exploration (0.0-1.0)
            
        Returns:
        --------
        list: List of parameter dictionaries
        """
        # Get all possible combinations
        all_combinations = list(self._generate_parameter_combinations())
        total_combinations = len(all_combinations)
        
        print(f"🎯 Smart Parameter Optimizer")
        print(f"   Total possible combinations: {total_combinations}")
        print(f"   Smart samples to test: {n_samples}")
        print(f"   Exploration factor: {exploration_factor}")
        
        # Strategy: Mix of random exploration and targeted sampling
        samples = []
        
        # 1. Always include some diverse baseline samples (spread across parameter space)
        baseline_samples = min(10, n_samples // 4)
        baseline_indices = [
            0,  # First combination
            total_combinations // 4,  # 25% through
            total_combinations // 2,  # 50% through  
            3 * total_combinations // 4,  # 75% through
            total_combinations - 1,  # Last combination
        ]
        
        # Add more diverse baseline samples if needed
        while len(baseline_indices) < baseline_samples:
            idx = random.randint(0, total_combinations - 1)
            if idx not in baseline_indices:
                baseline_indices.append(idx)
        
        for idx in baseline_indices[:baseline_samples]:
            samples.append(all_combinations[idx])
        
        # 2. Add random exploration samples
        exploration_samples = int(n_samples * exploration_factor)
        remaining_samples = n_samples - len(samples)
        exploration_samples = min(exploration_samples, remaining_samples)
        
        random_indices = random.sample(range(total_combinations), exploration_samples)
        for idx in random_indices:
            if all_combinations[idx] not in samples:
                samples.append(all_combinations[idx])
        
        # 3. Fill remaining with strategic samples (focus on different parameter regions)
        remaining = n_samples - len(samples)
        if remaining > 0:
            # Sample from different parameter regions
            for _ in range(remaining):
                # Create a combination that varies key parameters
                sample_params = {}
                for param, values in self.parameter_grid.items():
                    if param != 'hop_length_ratio':  # Skip ratio, will calculate hop_length
                        sample_params[param] = random.choice(values)
                
                # Calculate hop_length
                sample_params['hop_length'] = int(
                    sample_params['n_fft'] * random.choice(self.parameter_grid['hop_length_ratio'])
                )
                
                if sample_params not in samples:
                    samples.append(sample_params)
                
                if len(samples) >= n_samples:
                    break
        
        # Ensure we have exactly n_samples
        samples = samples[:n_samples]
        
        print(f"   Generated {len(samples)} diverse parameter combinations")
        return samples
    
    def run_smart_optimization(self, audio_pairs, n_iterations=50, save_results=True, save_spectrograms=False):
        """
        Run smart parameter optimization with adaptive exploration.
        
        Parameters:
        -----------
        audio_pairs : list
            List of audio pair dictionaries
        n_iterations : int
            Number of parameter combinations to test
        save_results : bool
            Whether to save results to files
        save_spectrograms : bool
            Whether to save spectrogram comparisons
            
        Returns:
        --------
        dict: Optimization results with best parameters and detailed analysis
        """
        print("🚀 Starting Smart Parameter Optimization")
        print(f"🎯 Testing {n_iterations} strategically selected combinations")
        print(f"🎵 Using {len(audio_pairs)} audio pairs for evaluation")
        print("=" * 80)
        
        # Generate smart parameter samples
        parameter_samples = self._generate_smart_parameter_samples(n_iterations)
        
        self.results = []
        
        # Create spectrograms directory if saving spectrograms
        if save_spectrograms:
            print("📸 Comprehensive spectrogram saving enabled!")
            print("   - Power Spectrograms (following Spectrogram_flow.png)")
            print("   - Mel-scale Spectrograms")
            print("   - Log-Mel Spectrograms") 
            print("   - LFCC feature plots")
            print("   - MFCC feature plots")
            print("   - All plots saved for each parameter combination tested")
            print("   - Saving to 'spectrograms_comparison' directory")
        
        # Track best results for adaptive sampling
        best_scores = []
        
        # Progress tracking with smart optimization
        with tqdm(total=len(parameter_samples), desc="Smart Parameter Optimization") as pbar:
            for i, params in enumerate(parameter_samples):
                result = self.evaluate_parameter_combination(params, audio_pairs, save_spectrograms=save_spectrograms)
                
                if result is not None:
                    self.results.append(result)
                    best_scores.append(result['final_score'])
                    
                    # Update best result
                    if result['final_score'] > self.best_score:
                        self.best_score = result['final_score']
                        self.best_params = params.copy()
                        
                        # Print improvement notification
                        print(f"\n🎉 New best score found at iteration {i+1}!")
                        print(f"   Score: {self.best_score:.4f}")
                        print(f"   Parameters: SR={params['sample_rate']} FFT={params['n_fft']} "
                              f"LFCC={params['n_lfcc']} MFCC={params['n_mfcc']} Mels={params['n_mels']}")
                
                # Adaptive exploration: if we've found good results, explore similar regions
                if i > 10 and i % 10 == 0:  # Every 10 iterations after initial exploration
                    recent_improvement = max(best_scores[-10:]) > (max(best_scores[:-10]) if len(best_scores) > 10 else 0)
                    if recent_improvement:
                        print(f"   🔍 Recent improvement detected - continuing exploration...")
                    
                pbar.set_postfix({
                    'Best Score': f"{self.best_score:.4f}",
                    'Current': f"{result['final_score']:.4f}" if result else "Failed",
                    'Avg Recent': f"{np.mean(best_scores[-5:]):.4f}" if len(best_scores) >= 5 else "N/A"
                })
                pbar.update(1)
        
        # Sort results by final score
        self.results.sort(key=lambda x: x['final_score'], reverse=True)
        
        print("\n🎉 Smart Optimization Complete!")
        print(f"✅ Evaluated {len(self.results)} successful combinations")
        print(f"🏆 Best score: {self.best_score:.4f}")
        print(f"📈 Score improvement: {self.best_score - best_scores[0]:.4f}" if best_scores else "N/A")
        
        # Show parameter diversity achieved
        self._analyze_parameter_diversity()
        
        # Save results
        if save_results:
            self._save_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'top_10': self.results[:10],
            'score_progression': best_scores
        }
    
    def _analyze_parameter_diversity(self):
        """Analyze diversity of parameters tested in smart optimization."""
        if not self.results:
            return
            
        print("\n🔍 PARAMETER DIVERSITY ANALYSIS")
        print("-" * 40)
        
        # Extract parameter values tested
        tested_params = {}
        for result in self.results:
            for param, value in result['params'].items():
                if param not in tested_params:
                    tested_params[param] = set()
                tested_params[param].add(value)
        
        # Show diversity for each parameter
        for param, values in tested_params.items():
            total_possible = len(self.parameter_grid.get(param, [param]))  # Handle hop_length
            if param == 'hop_length':
                # Skip hop_length as it's derived
                continue
                
            tested_count = len(values)
            coverage = tested_count / total_possible * 100
            
            print(f"{param:15s}: {tested_count}/{total_possible} values tested ({coverage:5.1f}% coverage)")
            if tested_count <= 6:  # Show values if not too many
                sorted_values = sorted(list(values))
                print(f"{'':17s}Values: {sorted_values}")
        
        print(f"\nTotal combinations tested: {len(self.results)}")
        print(f"Parameter space coverage: Diverse sampling across all dimensions")
    
    def load_audio_with_params(self, filepath, sample_rate):
        """Load audio with specific sample rate and preprocessing."""
        try:
            audio, sr = librosa.load(filepath, sr=sample_rate)
            
            # Apply pre-emphasis filter
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_multiple_spectrograms(self, audio, params):
        """
        Extract specific spectrograms following the flow diagram:
        1. Power Spectrogram (real-valued from raw spectrogram)
        2. Mel-scale Spectrogram (power)
        3. Log-Mel Spectrogram (dB-scaled)
        
        Returns:
        --------
        dict: Dictionary containing different spectrogram representations
        """
        spectrograms = {}
        
        # 1. Raw STFT -> Power Spectrogram (following flow diagram)
        stft = librosa.stft(
            audio, 
            n_fft=params['n_fft'], 
            hop_length=params['hop_length'], 
            window=params['window']
        )
        # Power Spectrogram: |STFT|^2 (real-valued)
        spectrograms['power_spectrogram'] = np.abs(stft) ** 2
        
        # 2. Mel-scale Spectrogram (following flow diagram)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=params['sample_rate'], 
            n_fft=params['n_fft'],
            hop_length=params['hop_length'], 
            window=params['window'],
            n_mels=params['n_mels']
        )
        spectrograms['mel_spectrogram'] = mel_spec
        
        # 3. Log-Mel Spectrogram (Mel power → dB). Converting to a
        #    logarithmic scale closely mimics human loudness perception
        #    and often accentuates the spectral smoothing introduced by
        #    generative vocoders, hence aiding fake detection.
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        spectrograms['log_mel_spectrogram'] = log_mel_spec
        
        return spectrograms
    
    def extract_features_with_params(self, audio, params):
        """Extract LFCC and MFCC features following the flow diagram."""
        features = {}
        
        # LFCC (from Power Spectrogram -> Apply filter bank -> Take logarithm -> Apply DCT)
        features['lfcc'] = self._extract_lfcc_with_params(audio, params)
        
        # MFCC (from Mel-scale Spectrogram -> Take logarithm -> Apply DCT)
        features['mfcc'] = self._extract_mfcc_with_params(audio, params)
        
        return features
    
    def _extract_lfcc_with_params(self, audio, params):
        """Extract LFCC with specific parameters.

        LFCCs provide a *linear* approximation of the spectrum that is
        particularly revealing for deep-fake artefacts that are often
        injected in lower (linear) frequency regions rather than on the
        perceptual Mel scale.  The number of coefficients (`n_lfcc`) is
        swept in the grid-search (40/60/80) to balance descriptive power
        against over-fitting.  A discrete cosine transform (DCT-II) is
        applied to decorrelate the log-energy filter-bank output, making
        the subsequent statistical comparison more stable.
        """
        stft = librosa.stft(
            audio, 
            n_fft=params['n_fft'], 
            hop_length=params['hop_length'], 
            window=params['window']
        )
        power_spectrum = np.abs(stft) ** 2
        
        filterbank = self._create_linear_filterbank(params['n_lfcc'], params)
        filter_energies = np.dot(filterbank, power_spectrum)
        filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
        log_energies = np.log(filter_energies)
        
        lfcc = dct(log_energies, type=2, axis=0, norm='ortho')
        return lfcc
    
    def _create_linear_filterbank(self, n_filters, params):
        """Create linear frequency filterbank with specific parameters."""
        n_fft_bins = params['n_fft'] // 2 + 1
        filterbank = np.zeros((n_filters, n_fft_bins))
        
        freq_points = np.linspace(0, params['sample_rate']/2, n_filters + 2)
        bin_points = np.floor((params['n_fft'] + 1) * freq_points / params['sample_rate']).astype(int)
        
        for i in range(1, n_filters + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            
            for j in range(left, center):
                if center != left:
                    filterbank[i-1, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if right != center:
                    filterbank[i-1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def _extract_mfcc_with_params(self, audio, params):
        """Extract MFCC with specific parameters.

        MFCCs mimic the human auditory system by mapping the linear
        spectrum onto a Mel scale *and* taking the logarithm of the
        filter-bank energies before the DCT.  Fake speech generation
        systems can inadvertently smooth or smear certain Mel bands; the
        MFCC representation therefore acts as a complementary view to
        LFCC.  The grid-search explores a small set of `n_mfcc` values
        (13/20/26), encompassing the traditional 13-coeff speech setup
        up to higher-dimensional variants that may capture more nuanced
        artefacts.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=params['sample_rate'], 
            n_fft=params['n_fft'],
            hop_length=params['hop_length'], 
            window=params['window'],
            n_mels=params['n_mels']  # Use parameter value for mel bands
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=params['n_mfcc'], dct_type=2, norm='ortho')
        return mfcc
    
    def _extract_log_mel_with_params(self, audio, params):
        """Extract Log-Mel spectrogram with specific parameters."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=params['sample_rate'], 
            n_fft=params['n_fft'],
            hop_length=params['hop_length'], 
            window=params['window'],
            n_mels=params['n_mels']
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def calculate_statistical_discrimination(self, features_real, features_fake):
        """
        Calculate statistical discrimination metrics between real and fake features.
        
        Returns:
        --------
        dict: Statistical discrimination scores
        """
        scores = {}
        
        for feature_type in features_real.keys():
            if feature_type not in features_fake:
                continue
                
            real_feat = features_real[feature_type]
            fake_feat = features_fake[feature_type]
            
            # Compute mean differences
            mean_real = np.mean(real_feat, axis=1)
            mean_fake = np.mean(fake_feat, axis=1)
            mean_diff = np.abs(mean_real - mean_fake)
            
            # Compute standard deviation ratios
            std_real = np.std(real_feat, axis=1)
            std_fake = np.std(fake_feat, axis=1)
            std_ratio = std_fake / (std_real + 1e-8)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((std_real**2 + std_fake**2) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-8)
            
            # Top discriminative coefficients
            top_5_indices = np.argsort(mean_diff)[-5:][::-1]
            top_5_scores = mean_diff[top_5_indices]
            
            scores[feature_type] = {
                'mean_absolute_difference': np.mean(mean_diff),
                'max_absolute_difference': np.max(mean_diff),
                'top_5_avg_difference': np.mean(top_5_scores),
                'avg_cohens_d': np.mean(cohens_d),
                'max_cohens_d': np.max(cohens_d),
                'std_ratio_avg': np.mean(std_ratio),
                'discriminative_coeffs': len(mean_diff[mean_diff > 0.5])
            }
        
        # Overall statistical score
        overall_score = np.mean([
            scores[ft]['top_5_avg_difference'] * scores[ft]['max_cohens_d'] 
            for ft in scores.keys()
        ])
        
        return scores, overall_score
    
    def calculate_visual_discriminability(self, spectrograms_real, spectrograms_fake):
        """
        Calculate visual discriminability metrics using computer vision techniques.
        
        Parameters:
        -----------
        spectrograms_real : dict
            Real audio spectrograms
        spectrograms_fake : dict
            Fake audio spectrograms
            
        Returns:
        --------
        dict: Visual discriminability scores
        """
        visual_scores = {}
        
        for spec_type in spectrograms_real.keys():
            if spec_type not in spectrograms_fake:
                continue
                
            real_spec = spectrograms_real[spec_type]
            fake_spec = spectrograms_fake[spec_type]
            
            # Normalize spectrograms to [0, 1] for visual metrics
            real_norm = self._normalize_spectrogram(real_spec)
            fake_norm = self._normalize_spectrogram(fake_spec)
            
            # Ensure same dimensions
            min_time = min(real_norm.shape[-1], fake_norm.shape[-1])
            real_norm = real_norm[..., :min_time]
            fake_norm = fake_norm[..., :min_time]
            
            scores = {}
            
            # 1. Structural Similarity Index (SSIM) - lower means more different
            try:
                ssim_score = ssim(real_norm, fake_norm, data_range=1.0)
                scores['ssim'] = 1.0 - ssim_score  # Invert so higher = more discriminable
            except:
                scores['ssim'] = 0.0
            
            # 2. Histogram Distance (Earth Mover's Distance)
            real_hist, _ = np.histogram(real_norm.flatten(), bins=50, density=True)
            fake_hist, _ = np.histogram(fake_norm.flatten(), bins=50, density=True)
            scores['histogram_distance'] = wasserstein_distance(
                np.arange(len(real_hist)), np.arange(len(fake_hist)), 
                real_hist, fake_hist
            )
            
            # 3. Edge Detection Differences (Sobel filter)
            real_edges = sobel(real_norm)
            fake_edges = sobel(fake_norm)
            edge_diff = np.mean(np.abs(real_edges - fake_edges))
            scores['edge_difference'] = edge_diff
            
            # 4. Pattern Regularity Analysis
            scores['pattern_regularity'] = self._calculate_pattern_regularity(real_norm, fake_norm)
            
            # 5. Frequency Band Contrast
            scores['frequency_contrast'] = self._calculate_frequency_contrast(real_norm, fake_norm)
            
            # 6. Temporal Structure Difference
            scores['temporal_structure'] = self._calculate_temporal_structure(real_norm, fake_norm)
            
            # 7. Mutual Information
            real_flat = (real_norm * 255).astype(int).flatten()
            fake_flat = (fake_norm * 255).astype(int).flatten()
            scores['mutual_information'] = mutual_info_score(real_flat, fake_flat)
            
            visual_scores[spec_type] = scores
        
        # Calculate overall visual discriminability score
        overall_visual_score = 0
        for spec_type, scores in visual_scores.items():
            spec_score = (
                scores['ssim'] * 0.25 +
                min(scores['histogram_distance'] / 10.0, 1.0) * 0.20 +
                min(scores['edge_difference'] * 10.0, 1.0) * 0.20 +
                scores['pattern_regularity'] * 0.15 +
                scores['frequency_contrast'] * 0.10 +
                scores['temporal_structure'] * 0.10
            )
            overall_visual_score += spec_score
        
        overall_visual_score /= len(visual_scores)
        
        return visual_scores, overall_visual_score
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _normalize_spectrogram(self, spectrogram):
        """Normalize spectrogram to [0, 1] range."""
        spec_min = np.min(spectrogram)
        spec_max = np.max(spectrogram)
        if spec_max - spec_min == 0:
            return np.zeros_like(spectrogram)
        return (spectrogram - spec_min) / (spec_max - spec_min)
    
    def _calculate_pattern_regularity(self, real_spec, fake_spec):
        """
        Calculate pattern regularity differences.
        Fake audio often shows unnatural smoothness or repetitive patterns.
        """
        # Calculate local variance
        real_var = np.var(real_spec, axis=1)
        fake_var = np.var(fake_spec, axis=1)
        
        # Calculate temporal gradient
        real_grad = np.abs(np.diff(real_spec, axis=1))
        fake_grad = np.abs(np.diff(fake_spec, axis=1))
        
        # Regularity score - higher means more difference in pattern complexity
        var_diff = np.mean(np.abs(real_var - fake_var))
        grad_diff = np.mean(np.abs(np.mean(real_grad, axis=1) - np.mean(fake_grad, axis=1)))
        
        return (var_diff + grad_diff) / 2.0
    
    def _calculate_frequency_contrast(self, real_spec, fake_spec):
        """Calculate contrast differences across frequency bands."""
        # Divide into frequency bands
        n_bands = min(8, real_spec.shape[0])
        band_size = real_spec.shape[0] // n_bands
        
        contrast_diffs = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, real_spec.shape[0])
            
            real_band = real_spec[start_idx:end_idx, :]
            fake_band = fake_spec[start_idx:end_idx, :]
            
            # Calculate contrast (standard deviation)
            real_contrast = np.std(real_band)
            fake_contrast = np.std(fake_band)
            
            contrast_diffs.append(abs(real_contrast - fake_contrast))
        
        return np.mean(contrast_diffs)
    
    def _calculate_temporal_structure(self, real_spec, fake_spec):
        """Calculate temporal structure differences."""
        # Calculate autocorrelation along time axis
        real_autocorr = []
        fake_autocorr = []
        
        for freq_bin in range(min(20, real_spec.shape[0])):  # Sample first 20 frequency bins
            real_signal = real_spec[freq_bin, :]
            fake_signal = fake_spec[freq_bin, :]
            
            # Autocorrelation
            real_auto = np.correlate(real_signal, real_signal, mode='full')
            fake_auto = np.correlate(fake_signal, fake_signal, mode='full')
            
            # Take center portion and normalize
            center = len(real_auto) // 2
            window = min(50, center)
            real_autocorr.append(real_auto[center-window:center+window])
            fake_autocorr.append(fake_auto[center-window:center+window])
        
        # Calculate difference in autocorrelation patterns
        real_autocorr = np.array(real_autocorr)
        fake_autocorr = np.array(fake_autocorr)
        
        return np.mean(np.abs(real_autocorr - fake_autocorr))
    
    def calculate_pattern_consistency(self, results_across_pairs):
        """
        Calculate consistency of discrimination across different audio pairs.
        
        Parameters:
        -----------
        results_across_pairs : list
            List of results from different audio pairs
            
        Returns:
        --------
        float: Consistency score (higher = more consistent)
        """
        if len(results_across_pairs) < 2:
            return 1.0
        
        # Extract discrimination scores across pairs
        stat_scores = [r['statistical_score'] for r in results_across_pairs]
        visual_scores = [r['visual_score'] for r in results_across_pairs]
        
        # Calculate coefficient of variation (lower = more consistent)
        stat_cv = np.std(stat_scores) / (np.mean(stat_scores) + 1e-8)
        visual_cv = np.std(visual_scores) / (np.mean(visual_scores) + 1e-8)
        
        # Convert to consistency score (higher = better)
        consistency_score = 1.0 / (1.0 + (stat_cv + visual_cv) / 2.0)
        
        return consistency_score
    
    def calculate_computational_efficiency(self, processing_time, params):
        """
        Calculate computational efficiency score.
        
        Parameters:
        -----------
        processing_time : float
            Time taken to process one audio pair (seconds)
        params : dict
            Parameter configuration
            
        Returns:
        --------
        float: Efficiency score (higher = more efficient)
        """
        # Normalize processing time (assuming 1 second as baseline)
        time_score = 1.0 / (1.0 + processing_time)
        
        # Parameter complexity penalty
        complexity_penalty = 0
        complexity_penalty += (params['sample_rate'] - 16000) / 32000  # Higher SR = more complex
        complexity_penalty += (params['n_fft'] - 512) / 3584  # Larger FFT = more complex
        complexity_penalty += (params['n_lfcc'] - 40) / 40  # More coefficients = more complex
        
        complexity_score = 1.0 / (1.0 + complexity_penalty)
        
        return (time_score + complexity_score) / 2.0
    
    def save_spectrograms_comparison(self, spectrograms_real, spectrograms_fake, params, pair_name, save_dir="spectrograms_comparison"):
        """
        Save individual spectrogram comparisons for visual analysis.
        
        Parameters:
        -----------
        spectrograms_real : dict
            Real audio spectrograms
        spectrograms_fake : dict
            Fake audio spectrograms
        params : dict
            Parameter configuration used
        pair_name : str
            Name of the audio pair
        save_dir : str
            Directory to save spectrograms
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create parameter string for filename
        param_str = f"sr{params['sample_rate']}_fft{params['n_fft']}_hop{params['hop_length']}_win{params['window']}"
        
        # Select spectrograms following the flow diagram
        spectrograms_to_save = {
            'power_spectrogram': 'Power Spectrogram',
            'mel_spectrogram': 'Mel-scale Spectrogram',
            'log_mel_spectrogram': 'Log-Mel Spectrogram'
        }
        
        for spec_type, spec_title in spectrograms_to_save.items():
            if spec_type in spectrograms_real and spec_type in spectrograms_fake:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    real_spec = spectrograms_real[spec_type]
                    fake_spec = spectrograms_fake[spec_type]
                    
                    # Normalize for visualization
                    real_norm = self._normalize_spectrogram(real_spec)
                    fake_norm = self._normalize_spectrogram(fake_spec)
                    
                    # Create time axes
                    time_real = np.linspace(0, real_spec.shape[1] * params['hop_length'] / params['sample_rate'], real_spec.shape[1])
                    time_fake = np.linspace(0, fake_spec.shape[1] * params['hop_length'] / params['sample_rate'], fake_spec.shape[1])
                    
                    # Plot real spectrogram
                    im1 = ax1.imshow(real_norm, aspect='auto', origin='lower', 
                                   extent=[0, time_real[-1], 0, real_spec.shape[0]], 
                                   cmap='magma')
                    ax1.set_title(f'{spec_title} - Real Audio\n{pair_name}', fontweight='bold')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Frequency Bin' if spec_type == 'power_spectrogram' else 'Mel Frequency Band')
                    plt.colorbar(im1, ax=ax1, label='Power' if spec_type == 'power_spectrogram' else 'Mel Power')
                    
                    # Plot fake spectrogram
                    im2 = ax2.imshow(fake_norm, aspect='auto', origin='lower',
                                   extent=[0, time_fake[-1], 0, fake_spec.shape[0]], 
                                   cmap='magma')
                    ax2.set_title(f'{spec_title} - Fake Audio\n{pair_name}', fontweight='bold')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Frequency Bin' if spec_type == 'power_spectrogram' else 'Mel Frequency Band')
                    plt.colorbar(im2, ax=ax2, label='Power' if spec_type == 'power_spectrogram' else 'Mel Power')
                    
                    # Add parameter info
                    param_text = f"SR: {params['sample_rate']}Hz, FFT: {params['n_fft']}, Hop: {params['hop_length']}, Win: {params['window']}"
                    fig.suptitle(f'{spec_title} Comparison - {param_text}', fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    filename = f"{pair_name}_{spec_type}_{param_str}.png"
                    filepath = os.path.join(save_dir, filename)
                    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    # ----------------------------------------------------------------
                    #  Additional figure – explicit Log-Mel Spectrogram comparison
                    # ----------------------------------------------------------------
                    try:
                        real_log_mel = spectrograms_real.get('log_mel_spectrogram')
                        fake_log_mel = spectrograms_fake.get('log_mel_spectrogram')
                        if real_log_mel is not None and fake_log_mel is not None:
                            fig_log, (axL1, axL2) = plt.subplots(1, 2, figsize=(16, 6))

                            imL1 = axL1.imshow(real_log_mel, aspect='auto', origin='lower',
                                               extent=[0, time_real[-1], 0, real_log_mel.shape[0]], cmap='viridis')
                            axL1.set_title('Log-Mel Spectrogram - Real Audio', fontweight='bold')
                            axL1.set_xlabel('Time (s)')
                            axL1.set_ylabel('Mel Frequency Band')
                            plt.colorbar(imL1, ax=axL1, label='Amplitude (dB)')

                            imL2 = axL2.imshow(fake_log_mel, aspect='auto', origin='lower',
                                               extent=[0, time_fake[-1], 0, fake_log_mel.shape[0]], cmap='viridis')
                            axL2.set_title('Log-Mel Spectrogram - Fake Audio', fontweight='bold')
                            axL2.set_xlabel('Time (s)')
                            axL2.set_ylabel('Mel Frequency Band')
                            plt.colorbar(imL2, ax=axL2, label='Amplitude (dB)')

                            plt.tight_layout()
                            filename_log = f"{pair_name}_logmel_{param_str}.png"
                            filepath_log = os.path.join(save_dir, filename_log)
                            plt.savefig(filepath_log, dpi=150, bbox_inches='tight', facecolor='white')
                            plt.close(fig_log)
                    except Exception as e:
                        print(f"Error saving Log-Mel spectrogram comparison: {e}")
                        plt.close()
                    
                except Exception as e:
                    print(f"Error saving {spec_type} spectrogram: {e}")
                    plt.close()

    def save_comprehensive_analysis(self, features_real, features_fake, spectrograms_real, spectrograms_fake, params, pair_name, save_dir="spectrograms_comparison"):
        """
        Save comprehensive analysis including spectrograms, LFCC, MFCC plots and statistical comparison.
        Following the Spectrogram_flow.png diagram, this saves all three spectrogram types and feature plots.
        
        Parameters:
        -----------
        features_real : dict
            Real audio features (LFCC, MFCC)
        features_fake : dict
            Fake audio features (LFCC, MFCC)
        spectrograms_real : dict
            Real audio spectrograms
        spectrograms_fake : dict
            Fake audio spectrograms
        params : dict
            Parameter configuration used
        pair_name : str
            Name of the audio pair
        save_dir : str
            Directory to save analysis
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create parameter string for filename
        param_str = f"sr{params['sample_rate']}_fft{params['n_fft']}_hop{params['hop_length']}_win{params['window']}_lfcc{params['n_lfcc']}_mfcc{params['n_mfcc']}_mels{params['n_mels']}"
        
        # 1. Save all three spectrograms from flow diagram (Power, Mel-scale, Log-Mel)
        self._save_all_spectrograms_flow(spectrograms_real, spectrograms_fake, params, pair_name, param_str, save_dir)
        
        # 2. Save LFCC and MFCC feature comparison plots
        self._save_feature_comparison(features_real, features_fake, params, pair_name, param_str, save_dir)
        
        # 3. Save statistical analysis
        self._save_statistical_analysis(features_real, features_fake, params, pair_name, param_str, save_dir)

    def _save_all_spectrograms_flow(self, spectrograms_real, spectrograms_fake, params, pair_name, param_str, save_dir):
        """
        Save all three spectrograms following the Spectrogram_flow.png diagram:
        1. Power Spectrogram (real-valued)
        2. Mel-scale Spectrogram 
        3. Log-Mel Spectrogram
        """
        try:
            # Create comprehensive 3x2 subplot for all spectrograms
            fig, axes = plt.subplots(3, 2, figsize=(18, 16))
            fig.suptitle(f'Complete Spectrogram Flow Analysis: {pair_name}\n{param_str}', fontsize=16, fontweight='bold')
            
            # Time axis for plotting
            time_real = np.linspace(0, spectrograms_real['power_spectrogram'].shape[1] * params['hop_length'] / params['sample_rate'], 
                                  spectrograms_real['power_spectrogram'].shape[1])
            time_fake = np.linspace(0, spectrograms_fake['power_spectrogram'].shape[1] * params['hop_length'] / params['sample_rate'], 
                                  spectrograms_fake['power_spectrogram'].shape[1])
            
            # 1. Power Spectrogram (convert to dB for visualization)
            real_power = spectrograms_real['power_spectrogram']
            fake_power = spectrograms_fake['power_spectrogram']
            real_power_db = librosa.power_to_db(real_power, ref=np.max)
            fake_power_db = librosa.power_to_db(fake_power, ref=np.max)
            
            im1 = axes[0,0].imshow(real_power_db, aspect='auto', origin='lower', 
                                 extent=[0, time_real[-1], 0, real_power_db.shape[0]], cmap='magma')
            axes[0,0].set_title('Power Spectrogram - Real Audio', fontweight='bold')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Frequency Bin')
            plt.colorbar(im1, ax=axes[0,0], label='Power (dB)')
            
            im2 = axes[0,1].imshow(fake_power_db, aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, fake_power_db.shape[0]], cmap='magma')
            axes[0,1].set_title('Power Spectrogram - Fake Audio', fontweight='bold')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Frequency Bin')
            plt.colorbar(im2, ax=axes[0,1], label='Power (dB)')
            
            # 2. Mel-scale Spectrogram (convert to dB for visualization)
            real_mel = spectrograms_real['mel_spectrogram']
            fake_mel = spectrograms_fake['mel_spectrogram']
            real_mel_db = librosa.power_to_db(real_mel, ref=np.max)
            fake_mel_db = librosa.power_to_db(fake_mel, ref=np.max)
            
            im3 = axes[1,0].imshow(real_mel_db, aspect='auto', origin='lower',
                                 extent=[0, time_real[-1], 0, real_mel_db.shape[0]], cmap='viridis')
            axes[1,0].set_title('Mel-scale Spectrogram - Real Audio', fontweight='bold')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Mel Frequency Band')
            plt.colorbar(im3, ax=axes[1,0], label='Mel Power (dB)')
            
            im4 = axes[1,1].imshow(fake_mel_db, aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, fake_mel_db.shape[0]], cmap='viridis')
            axes[1,1].set_title('Mel-scale Spectrogram - Fake Audio', fontweight='bold')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Mel Frequency Band')
            plt.colorbar(im4, ax=axes[1,1], label='Mel Power (dB)')
            
            # 3. Log-Mel Spectrogram (already in dB)
            real_log_mel = spectrograms_real['log_mel_spectrogram']
            fake_log_mel = spectrograms_fake['log_mel_spectrogram']
            
            im5 = axes[2,0].imshow(real_log_mel, aspect='auto', origin='lower',
                                 extent=[0, time_real[-1], 0, real_log_mel.shape[0]], cmap='plasma')
            axes[2,0].set_title('Log-Mel Spectrogram - Real Audio', fontweight='bold')
            axes[2,0].set_xlabel('Time (s)')
            axes[2,0].set_ylabel('Mel Frequency Band')
            plt.colorbar(im5, ax=axes[2,0], label='Amplitude (dB)')
            
            im6 = axes[2,1].imshow(fake_log_mel, aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, fake_log_mel.shape[0]], cmap='plasma')
            axes[2,1].set_title('Log-Mel Spectrogram - Fake Audio', fontweight='bold')
            axes[2,1].set_xlabel('Time (s)')
            axes[2,1].set_ylabel('Mel Frequency Band')
            plt.colorbar(im6, ax=axes[2,1], label='Amplitude (dB)')
            
            plt.tight_layout()
            
            # Save complete spectrogram flow diagram
            filename = f"{pair_name}_spectrogram_flow_{param_str}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"Error saving spectrogram flow diagram: {e}")
            plt.close()

    def _save_spectrogram_comparison(self, spectrograms_real, spectrograms_fake, params, pair_name, param_str, save_dir):
        """Save Power Spectrogram and Mel-scale Spectrogram comparison."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Spectrogram Analysis: {pair_name} - {param_str}', fontsize=16, fontweight='bold')
            
            # Power Spectrogram
            real_power = spectrograms_real['power_spectrogram']
            fake_power = spectrograms_fake['power_spectrogram']
            
            # Convert to dB for visualization
            real_power_db = librosa.power_to_db(real_power, ref=np.max)
            fake_power_db = librosa.power_to_db(fake_power, ref=np.max)
            
            time_real = np.linspace(0, real_power.shape[1] * params['hop_length'] / params['sample_rate'], real_power.shape[1])
            time_fake = np.linspace(0, fake_power.shape[1] * params['hop_length'] / params['sample_rate'], fake_power.shape[1])
            
            # Plot Power Spectrograms
            im1 = axes[0,0].imshow(real_power_db, aspect='auto', origin='lower', 
                                 extent=[0, time_real[-1], 0, real_power_db.shape[0]], cmap='magma')
            axes[0,0].set_title('Power Spectrogram - Real Audio', fontweight='bold')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Frequency Bin')
            plt.colorbar(im1, ax=axes[0,0], label='Power (dB)')
            
            im2 = axes[0,1].imshow(fake_power_db, aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, fake_power_db.shape[0]], cmap='magma')
            axes[0,1].set_title('Power Spectrogram - Fake Audio', fontweight='bold')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Frequency Bin')
            plt.colorbar(im2, ax=axes[0,1], label='Power (dB)')
            
            # Mel-scale Spectrogram
            real_mel = spectrograms_real['mel_spectrogram']
            fake_mel = spectrograms_fake['mel_spectrogram']
            
            # Convert to dB for visualization
            real_mel_db = librosa.power_to_db(real_mel, ref=np.max)
            fake_mel_db = librosa.power_to_db(fake_mel, ref=np.max)
            
            # Plot Mel Spectrograms
            im3 = axes[1,0].imshow(real_mel_db, aspect='auto', origin='lower',
                                 extent=[0, time_real[-1], 0, real_mel_db.shape[0]], cmap='viridis')
            axes[1,0].set_title('Mel-scale Spectrogram - Real Audio', fontweight='bold')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Mel Frequency Band')
            plt.colorbar(im3, ax=axes[1,0], label='Mel Power (dB)')
            
            im4 = axes[1,1].imshow(fake_mel_db, aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, fake_mel_db.shape[0]], cmap='viridis')
            axes[1,1].set_title('Mel-scale Spectrogram - Fake Audio', fontweight='bold')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Mel Frequency Band')
            plt.colorbar(im4, ax=axes[1,1], label='Mel Power (dB)')
            
            plt.tight_layout()
            
            # Save figure
            filename = f"{pair_name}_spectrograms_{param_str}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # ----------------------------------------------------------------
            #  Additional figure – explicit Log-Mel Spectrogram comparison
            # ----------------------------------------------------------------
            try:
                real_log_mel = spectrograms_real.get('log_mel_spectrogram')
                fake_log_mel = spectrograms_fake.get('log_mel_spectrogram')
                if real_log_mel is not None and fake_log_mel is not None:
                    fig_log, (axL1, axL2) = plt.subplots(1, 2, figsize=(16, 6))

                    imL1 = axL1.imshow(real_log_mel, aspect='auto', origin='lower',
                                       extent=[0, time_real[-1], 0, real_log_mel.shape[0]], cmap='viridis')
                    axL1.set_title('Log-Mel Spectrogram - Real Audio', fontweight='bold')
                    axL1.set_xlabel('Time (s)')
                    axL1.set_ylabel('Mel Frequency Band')
                    plt.colorbar(imL1, ax=axL1, label='Amplitude (dB)')

                    imL2 = axL2.imshow(fake_log_mel, aspect='auto', origin='lower',
                                       extent=[0, time_fake[-1], 0, fake_log_mel.shape[0]], cmap='viridis')
                    axL2.set_title('Log-Mel Spectrogram - Fake Audio', fontweight='bold')
                    axL2.set_xlabel('Time (s)')
                    axL2.set_ylabel('Mel Frequency Band')
                    plt.colorbar(imL2, ax=axL2, label='Amplitude (dB)')

                    plt.tight_layout()
                    filename_log = f"{pair_name}_logmel_{param_str}.png"
                    filepath_log = os.path.join(save_dir, filename_log)
                    plt.savefig(filepath_log, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig_log)
            except Exception as e:
                print(f"Error saving Log-Mel spectrogram comparison: {e}")
                plt.close()
            
        except Exception as e:
            print(f"Error saving spectrogram comparison: {e}")
            plt.close()

    def _save_feature_comparison(self, features_real, features_fake, params, pair_name, param_str, save_dir):
        """Save LFCC and MFCC feature comparison plots with parameter-specific dimensions."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Cepstral Features Analysis: {pair_name}\nLFCC: {params["n_lfcc"]}, MFCC: {params["n_mfcc"]}\n{param_str}', 
                        fontsize=14, fontweight='bold')
            
            # LFCC - use the actual number of coefficients from parameters
            real_lfcc = features_real['lfcc']
            fake_lfcc = features_fake['lfcc']
            
            # Display all LFCC coefficients (up to n_lfcc parameter)
            n_lfcc_display = min(params['n_lfcc'], real_lfcc.shape[0])
            
            time_real = np.linspace(0, real_lfcc.shape[1] * params['hop_length'] / params['sample_rate'], real_lfcc.shape[1])
            time_fake = np.linspace(0, fake_lfcc.shape[1] * params['hop_length'] / params['sample_rate'], fake_lfcc.shape[1])
            
            # Plot LFCC - show all coefficients for this parameter combination
            im1 = axes[0,0].imshow(real_lfcc[:n_lfcc_display], aspect='auto', origin='lower',
                                 extent=[0, time_real[-1], 0, n_lfcc_display], cmap='coolwarm')
            axes[0,0].set_title(f'LFCC ({n_lfcc_display} coeffs) - Real Audio', fontweight='bold')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('LFCC Coefficient')
            plt.colorbar(im1, ax=axes[0,0], label='Amplitude')
            
            im2 = axes[0,1].imshow(fake_lfcc[:n_lfcc_display], aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, n_lfcc_display], cmap='coolwarm')
            axes[0,1].set_title(f'LFCC ({n_lfcc_display} coeffs) - Fake Audio', fontweight='bold')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('LFCC Coefficient')
            plt.colorbar(im2, ax=axes[0,1], label='Amplitude')
            
            # MFCC - use the actual number of coefficients from parameters
            real_mfcc = features_real['mfcc']
            fake_mfcc = features_fake['mfcc']
            n_mfcc_display = min(params['n_mfcc'], real_mfcc.shape[0])
            
            # Plot MFCC - show all coefficients for this parameter combination
            im3 = axes[1,0].imshow(real_mfcc[:n_mfcc_display], aspect='auto', origin='lower',
                                 extent=[0, time_real[-1], 0, n_mfcc_display], cmap='plasma')
            axes[1,0].set_title(f'MFCC ({n_mfcc_display} coeffs) - Real Audio', fontweight='bold')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('MFCC Coefficient')
            plt.colorbar(im3, ax=axes[1,0], label='Amplitude')
            
            im4 = axes[1,1].imshow(fake_mfcc[:n_mfcc_display], aspect='auto', origin='lower',
                                 extent=[0, time_fake[-1], 0, n_mfcc_display], cmap='plasma')
            axes[1,1].set_title(f'MFCC ({n_mfcc_display} coeffs) - Fake Audio', fontweight='bold')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('MFCC Coefficient')
            plt.colorbar(im4, ax=axes[1,1], label='Amplitude')
            
            plt.tight_layout()
            
            # Save comprehensive feature comparison
            filename = f"{pair_name}_features_{param_str}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Also save individual LFCC and MFCC plots for detailed analysis
            self._save_individual_feature_plots(features_real, features_fake, params, pair_name, param_str, save_dir)
            
        except Exception as e:
            print(f"Error saving feature comparison: {e}")
            plt.close()

    def _save_individual_feature_plots(self, features_real, features_fake, params, pair_name, param_str, save_dir):
        """Save individual LFCC and MFCC plots for detailed analysis."""
        try:
            # Individual LFCC plot
            fig_lfcc, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig_lfcc.suptitle(f'LFCC Detailed Analysis: {pair_name} - {params["n_lfcc"]} coefficients\n{param_str}', 
                             fontsize=14, fontweight='bold')
            
            real_lfcc = features_real['lfcc']
            fake_lfcc = features_fake['lfcc']
            n_lfcc_display = min(params['n_lfcc'], real_lfcc.shape[0])
            
            time_real = np.linspace(0, real_lfcc.shape[1] * params['hop_length'] / params['sample_rate'], real_lfcc.shape[1])
            time_fake = np.linspace(0, fake_lfcc.shape[1] * params['hop_length'] / params['sample_rate'], fake_lfcc.shape[1])
            
            im1 = ax1.imshow(real_lfcc[:n_lfcc_display], aspect='auto', origin='lower',
                           extent=[0, time_real[-1], 0, n_lfcc_display], cmap='coolwarm')
            ax1.set_title(f'LFCC - Real Audio ({n_lfcc_display} coeffs)', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('LFCC Coefficient')
            plt.colorbar(im1, ax=ax1, label='Amplitude')
            
            im2 = ax2.imshow(fake_lfcc[:n_lfcc_display], aspect='auto', origin='lower',
                           extent=[0, time_fake[-1], 0, n_lfcc_display], cmap='coolwarm')
            ax2.set_title(f'LFCC - Fake Audio ({n_lfcc_display} coeffs)', fontweight='bold')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('LFCC Coefficient')
            plt.colorbar(im2, ax=ax2, label='Amplitude')
            
            plt.tight_layout()
            filename_lfcc = f"{pair_name}_lfcc_detailed_{param_str}.png"
            filepath_lfcc = os.path.join(save_dir, filename_lfcc)
            plt.savefig(filepath_lfcc, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Individual MFCC plot
            fig_mfcc, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
            fig_mfcc.suptitle(f'MFCC Detailed Analysis: {pair_name} - {params["n_mfcc"]} coefficients\n{param_str}', 
                             fontsize=14, fontweight='bold')
            
            real_mfcc = features_real['mfcc']
            fake_mfcc = features_fake['mfcc']
            n_mfcc_display = min(params['n_mfcc'], real_mfcc.shape[0])
            
            im3 = ax3.imshow(real_mfcc[:n_mfcc_display], aspect='auto', origin='lower',
                           extent=[0, time_real[-1], 0, n_mfcc_display], cmap='plasma')
            ax3.set_title(f'MFCC - Real Audio ({n_mfcc_display} coeffs)', fontweight='bold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('MFCC Coefficient')
            plt.colorbar(im3, ax=ax3, label='Amplitude')
            
            im4 = ax4.imshow(fake_mfcc[:n_mfcc_display], aspect='auto', origin='lower',
                           extent=[0, time_fake[-1], 0, n_mfcc_display], cmap='plasma')
            ax4.set_title(f'MFCC - Fake Audio ({n_mfcc_display} coeffs)', fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('MFCC Coefficient')
            plt.colorbar(im4, ax=ax4, label='Amplitude')
            
            plt.tight_layout()
            filename_mfcc = f"{pair_name}_mfcc_detailed_{param_str}.png"
            filepath_mfcc = os.path.join(save_dir, filename_mfcc)
            plt.savefig(filepath_mfcc, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"Error saving individual feature plots: {e}")
            plt.close()

    def _save_statistical_analysis(self, features_real, features_fake, params, pair_name, param_str, save_dir):
        """Save detailed statistical analysis as text file."""
        try:
            filename = f"{pair_name}_statistics_{param_str}.txt"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Statistical Analysis: {pair_name}\n")
                f.write(f"Parameters: {param_str}\n")
                f.write("=" * 60 + "\n\n")
                
                # Calculate statistical discrimination
                stat_scores, _ = self.calculate_statistical_discrimination(features_real, features_fake)
                
                for feature_type, scores in stat_scores.items():
                    f.write(f"{feature_type.upper()} Statistical Analysis:\n")
                    f.write("-" * 40 + "\n")
                    
                    real_feat = features_real[feature_type]
                    fake_feat = features_fake[feature_type]
                    
                    # Basic statistics
                    f.write(f"Real Audio - Mean: {np.mean(real_feat):.6f}, Std: {np.std(real_feat):.6f}\n")
                    f.write(f"Fake Audio - Mean: {np.mean(fake_feat):.6f}, Std: {np.std(fake_feat):.6f}\n")
                    f.write(f"Mean Absolute Difference: {scores['mean_absolute_difference']:.6f}\n")
                    f.write(f"Max Absolute Difference: {scores['max_absolute_difference']:.6f}\n")
                    f.write(f"Top 5 Avg Difference: {scores['top_5_avg_difference']:.6f}\n")
                    f.write(f"Average Cohen's d: {scores['avg_cohens_d']:.6f}\n")
                    f.write(f"Max Cohen's d: {scores['max_cohens_d']:.6f}\n")
                    f.write(f"Std Ratio (fake/real): {scores['std_ratio_avg']:.6f}\n")
                    f.write(f"Discriminative Coeffs (>0.5): {scores['discriminative_coeffs']}\n")
                    f.write("\n")
                
        except Exception as e:
            print(f"Error saving statistical analysis: {e}")

    def evaluate_parameter_combination(self, params, audio_pairs, save_spectrograms=False):
        """
        Evaluate a single parameter combination across all audio pairs.
        
        Parameters:
        -----------
        params : dict
            Parameter configuration to evaluate
        audio_pairs : list
            List of audio pair dictionaries with 'real' and 'fake' paths
        save_spectrograms : bool
            Whether to save spectrogram comparisons
            
        Returns:
        --------
        dict: Comprehensive evaluation results
        """
        start_time = time.time()
        pair_results = []
        
        for pair_info in audio_pairs:
            try:
                # Load audio files
                audio_real = self.load_audio_with_params(pair_info['real'], params['sample_rate'])
                audio_fake = self.load_audio_with_params(pair_info['fake'], params['sample_rate'])
                
                if audio_real is None or audio_fake is None:
                    continue
                
                # Extract features
                features_real = self.extract_features_with_params(audio_real, params)
                features_fake = self.extract_features_with_params(audio_fake, params)
                
                # Extract spectrograms
                spectrograms_real = self.extract_multiple_spectrograms(audio_real, params)
                spectrograms_fake = self.extract_multiple_spectrograms(audio_fake, params)
                
                # Save comprehensive analysis if requested
                if save_spectrograms:
                    self.save_comprehensive_analysis(
                        features_real, features_fake, spectrograms_real, spectrograms_fake, 
                        params, pair_info.get('pair_name', 'unknown')
                    )
                
                # Calculate scores
                stat_scores, stat_overall = self.calculate_statistical_discrimination(
                    features_real, features_fake
                )
                visual_scores, visual_overall = self.calculate_visual_discriminability(
                    spectrograms_real, spectrograms_fake
                )
                
                pair_results.append({
                    'pair_name': pair_info.get('pair_name', 'unknown'),
                    'statistical_score': stat_overall,
                    'visual_score': visual_overall,
                    'statistical_details': stat_scores,
                    'visual_details': visual_scores
                })
                
            except Exception as e:
                print(f"Error processing pair {pair_info.get('pair_name', 'unknown')}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        if not pair_results:
            return None
        
        # Calculate overall metrics
        avg_statistical = np.mean([r['statistical_score'] for r in pair_results])
        avg_visual = np.mean([r['visual_score'] for r in pair_results])
        consistency = self.calculate_pattern_consistency(pair_results)
        
        # Calculate weighted final score (removed computational efficiency)
        final_score = (
            self.scoring_weights['statistical_discrimination'] * avg_statistical +
            self.scoring_weights['visual_discriminability'] * avg_visual +
            self.scoring_weights['pattern_consistency'] * consistency
        )
        
        return {
            'params': params,
            'final_score': final_score,
            'statistical_score': avg_statistical,
            'visual_score': avg_visual,
            'consistency_score': consistency,
            'processing_time': processing_time,
            'pair_results': pair_results
        }
    
    def run_optimization(self, audio_pairs, max_combinations=None, save_results=True, save_spectrograms=False):
        """
        Run the complete parameter optimization process.
        
        Parameters:
        -----------
        audio_pairs : list
            List of audio pair dictionaries
        max_combinations : int, optional
            Maximum number of combinations to test (for debugging)
        save_results : bool
            Whether to save results to files
        save_spectrograms : bool
            Whether to save spectrogram comparisons for each parameter combination
            
        Returns:
        --------
        dict: Optimization results with best parameters and detailed analysis
        """
        print("🚀 Starting Parameter Optimization Process")
        print(f"📊 Testing {min(self._count_combinations(), max_combinations or float('inf'))} combinations")
        print(f"🎵 Using {len(audio_pairs)} audio pairs for evaluation")
        print("=" * 80)
        
        combinations = list(self._generate_parameter_combinations())
        if max_combinations:
            combinations = combinations[:max_combinations]
        
        self.results = []
        
        # Create spectrograms directory if saving spectrograms
        if save_spectrograms:
            print("📸 Comprehensive spectrogram saving enabled!")
            print("   - Power Spectrograms (following Spectrogram_flow.png)")
            print("   - Mel-scale Spectrograms")
            print("   - Log-Mel Spectrograms") 
            print("   - LFCC feature plots")
            print("   - MFCC feature plots")
            print("   - All plots saved for each parameter combination tested")
            print("   - Saving to 'spectrograms_comparison' directory")
        
        # Progress tracking
        with tqdm(total=len(combinations), desc="Optimizing Parameters") as pbar:
            for i, params in enumerate(combinations):
                result = self.evaluate_parameter_combination(params, audio_pairs, save_spectrograms=save_spectrograms)
                
                if result is not None:
                    self.results.append(result)
                    
                    # Update best result
                    if result['final_score'] > self.best_score:
                        self.best_score = result['final_score']
                        self.best_params = params.copy()
                
                pbar.set_postfix({
                    'Best Score': f"{self.best_score:.4f}",
                    'Current': f"{result['final_score']:.4f}" if result else "Failed"
                })
                pbar.update(1)
        
        # Sort results by final score
        self.results.sort(key=lambda x: x['final_score'], reverse=True)
        
        print("\n🎉 Optimization Complete!")
        print(f"✅ Evaluated {len(self.results)} successful combinations")
        print(f"🏆 Best score: {self.best_score:.4f}")
        
        # Save results
        if save_results:
            self._save_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'top_10': self.results[:10]
        }
    
    def _save_results(self):
        """Save optimization results to files."""
        timestamp = int(time.time())
        
        # Save detailed results
        results_file = f"optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for result in self.results:
                serializable_result = self._make_json_serializable(result)
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Detailed results saved to: {results_file}")
        
        # Save summary CSV
        summary_file = f"optimization_summary_{timestamp}.csv"
        summary_data = []
        for result in self.results:
            row = result['params'].copy()
            row.update({
                'final_score': result['final_score'],
                'statistical_score': result['statistical_score'],
                'visual_score': result['visual_score'],
                'consistency_score': result['consistency_score'],
                'processing_time': result['processing_time']
            })
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        print(f"📋 Summary CSV saved to: {summary_file}")
        
        # Save best parameters
        best_params_file = f"best_parameters_{timestamp}.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': float(self.best_score),
                'optimization_settings': {
                    'scoring_weights': self.scoring_weights,
                    'parameter_grid': self.parameter_grid
                }
            }, f, indent=2)
        print(f"🏆 Best parameters saved to: {best_params_file}")
    
    def analyze_results(self):
        """Analyze and visualize optimization results."""
        if not self.results:
            print("❌ No results to analyze. Run optimization first.")
            return
        
        print("\n📈 PARAMETER OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        # Best parameters analysis
        print(f"\n🏆 BEST PARAMETERS (Score: {self.best_score:.4f})")
        print("-" * 40)
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Top 5 results
        print(f"\n🥇 TOP 5 PARAMETER COMBINATIONS")
        print("-" * 40)
        for i, result in enumerate(self.results[:5]):
            print(f"{i+1}. Score: {result['final_score']:.4f}")
            print(f"   Sample Rate: {result['params']['sample_rate']} Hz")
            print(f"   FFT Size: {result['params']['n_fft']}")
            print(f"   Hop Length: {result['params']['hop_length']}")
            print(f"   Window: {result['params']['window']}")
            print()
        
        # Parameter impact analysis
        self._analyze_parameter_impact()
        
        # Score distribution analysis
        self._analyze_score_distribution()
    
    def _analyze_parameter_impact(self):
        """Analyze the impact of individual parameters on performance."""
        print("\n🔍 PARAMETER IMPACT ANALYSIS")
        print("-" * 40)
        
        # Create DataFrame for analysis
        data = []
        for result in self.results:
            row = result['params'].copy()
            row['final_score'] = result['final_score']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Analyze each parameter
        for param in ['sample_rate', 'n_fft', 'hop_length', 'window']:
            if param in df.columns:
                param_analysis = df.groupby(param)['final_score'].agg(['mean', 'std', 'count'])
                print(f"\n{param.upper()}:")
                for value, stats in param_analysis.iterrows():
                    print(f"  {value}: avg={stats['mean']:.4f} (±{stats['std']:.4f}), n={stats['count']}")
    
    def _analyze_score_distribution(self):
        """Analyze distribution of scores across different metrics."""
        print("\n📊 SCORE DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        final_scores = [r['final_score'] for r in self.results]
        stat_scores = [r['statistical_score'] for r in self.results]
        visual_scores = [r['visual_score'] for r in self.results]
        
        print(f"Final Score    - Mean: {np.mean(final_scores):.4f}, Std: {np.std(final_scores):.4f}")
        print(f"Statistical   - Mean: {np.mean(stat_scores):.4f}, Std: {np.std(stat_scores):.4f}")
        print(f"Visual        - Mean: {np.mean(visual_scores):.4f}, Std: {np.std(visual_scores):.4f}")
        
        print(f"\nTop 10% threshold: {np.percentile(final_scores, 90):.4f}")
        print(f"Bottom 10% threshold: {np.percentile(final_scores, 10):.4f}")


def main():
    """
    Main function to run the parameter optimization process.
    """
    print("🎯 OPTIMAL AUDIO DEEPFAKE DETECTION PARAMETERS")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = OptimalDeepfakeAnalyzer()
    
    # Define audio pairs for testing
    base_path = "Audio Samples"
    audio_pairs = [
        {
            'real': os.path.join(base_path, "Pair_1", "Pair_1_R.wav"),
            'fake': os.path.join(base_path, "Pair_1", "Pair_1_F.wav"),
            'pair_name': 'Pair_1'
        },
        {
            'real': os.path.join(base_path, "Pair_2", "Pair_2_R.wav"),
            'fake': os.path.join(base_path, "Pair_2", "Pair2_F.wav"),
            'pair_name': 'Pair_2'
        }
    ]
    
    # Run smart optimization with strategic parameter sampling
    print("🔬 Starting smart optimization process...")
    print("🎯 Using intelligent parameter exploration instead of exhaustive grid search")
    print("💡 This approach tests diverse parameter combinations efficiently")
    
    results = optimizer.run_smart_optimization(
        audio_pairs=audio_pairs,
        n_iterations=50,  # Test 50 strategically selected combinations
        save_results=True,
        save_spectrograms=True  # Enable comprehensive analysis saving
    )
    
    # Analyze results
    optimizer.analyze_results()
    
    # Show optimization progress
    if 'score_progression' in results:
        print(f"\n📈 OPTIMIZATION PROGRESS")
        print("-" * 40)
        scores = results['score_progression']
        print(f"Initial score: {scores[0]:.4f}")
        print(f"Final best score: {scores[-1] if scores else 0:.4f}")
        print(f"Total improvement: {(scores[-1] - scores[0]):.4f}" if len(scores) > 0 else "N/A")
        
        # Show score progression in chunks
        chunk_size = 10
        for i in range(0, len(scores), chunk_size):
            chunk = scores[i:i+chunk_size]
            chunk_max = max(chunk)
            chunk_avg = np.mean(chunk)
            print(f"Iterations {i+1:2d}-{min(i+chunk_size, len(scores)):2d}: "
                  f"max={chunk_max:.4f}, avg={chunk_avg:.4f}")
    
    # Generate final recommendations
    print("\n💡 FINAL RECOMMENDATIONS")
    print("=" * 60)
    print(f"🎯 Optimal parameters found with score: {results['best_score']:.4f}")
    print("\nThese parameters provide the best balance of:")
    print("  ✅ Statistical discrimination between real and fake audio")
    print("  ✅ Visual discriminability in spectrograms")
    print("  ✅ Consistency across different audio pairs")
    
    print(f"\n🔧 RECOMMENDED CONFIGURATION:")
    print("-" * 30)
    if results['best_params']:
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
    else:
        print("  No valid parameters found - all combinations failed")
    
    print("\n🎉 Optimization complete! Check the saved files for detailed results.")


if __name__ == "__main__":
    main()