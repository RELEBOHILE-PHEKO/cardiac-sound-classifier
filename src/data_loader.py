"""
Dataset loader that integrates with my AudioPreprocessor.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from preprocessing import AudioPreprocessor


@dataclass
class CardiacDatasetLoader:
    """
    Loads the PhysioNet heart sound dataset and prepares it for training.
    This plugs directly into the AudioPreprocessor I already built.
    """
    
    train_dir: str = 'data/train/training'
    test_dir: str = 'data/validation'
    preprocessor: Optional[AudioPreprocessor] = None
    
    def __post_init__(self):
        # Set up folder paths
        self.train_path = Path(self.train_dir)
        # Support both new and legacy layout: prefer `data/validation`, fallback to `data/test/validation`
        self.test_path = Path(self.test_dir)
        if not self.test_path.exists():
            legacy = Path('data') / 'test' / 'validation'
            if legacy.exists():
                print("Using legacy validation path: data/test/validation")
                self.test_path = legacy
        
        # If no preprocessor is passed in, I just make one on the fly
        if self.preprocessor is None:
            self.preprocessor = AudioPreprocessor(
                target_sample_rate=4000,
                segment_seconds=5.0,
                n_mels=128
            )
        
        # PhysioNet splits its training data into multiple mini-folders
        # (training-a, training-b, ..., training-f)
        # So here I find all those folders automatically.
        self.train_subsets = [
            d for d in self.train_path.iterdir()
            if d.is_dir() and d.name.startswith('training-')
        ]
        
        print(f"Found {len(self.train_subsets)} training subsets:")
        for subset in self.train_subsets:
            print(f"   - {subset.name}")
    
    def load_labels_from_subset(self, subset_dir: Path) -> pd.DataFrame:
        """
        Load REFERENCE.csv from one subset. This file maps recording → label.
        """
        reference_file = subset_dir / 'REFERENCE.csv'
        
        # If the subset has no labels file, I skip it
        if not reference_file.exists():
            print(f"   Warning: No REFERENCE.csv in {subset_dir.name}")
            return pd.DataFrame()
        
        # Read the labels (PhysioNet format: filename,label)
        df = pd.read_csv(reference_file, names=['filename', 'label'])
        
        # Point each filename to its .wav file inside the same folder
        df['wav_path'] = df['filename'].apply(lambda x: str(subset_dir / f"{x}.wav"))
        
        # Keep track of which subset this came from
        df['subset'] = subset_dir.name
        
        # Check if the wav file actually exists (PhysioNet sometimes has missing files)
        df['file_exists'] = df['wav_path'].apply(lambda p: Path(p).exists())
        
        missing = (~df['file_exists']).sum()
        if missing > 0:
            print(f"   Warning: {missing} files missing in {subset_dir.name}")
            df = df[df['file_exists']]
        
        return df[['filename', 'label', 'wav_path', 'subset']]
    
    def load_all_training_labels(self) -> pd.DataFrame:
        """
        Load labels from all training subsets and merge into one dataframe.
        """
        print("\nLoading training labels...")
        all_labels = []
        
        for subset_dir in self.train_subsets:
            df = self.load_labels_from_subset(subset_dir)
            if not df.empty:
                all_labels.append(df)
                print(f"   Loaded {len(df)} files from {subset_dir.name}")
        
        if not all_labels:
            raise ValueError("No labels found in any training subset!")
        
        # Combine everything into one big table
        combined_df = pd.concat(all_labels, ignore_index=True)
        
        # PhysioNet:  1 = abnormal,   -1 = normal
        # My model:   1 = abnormal,    0 = normal
        combined_df['label_binary'] = (combined_df['label'] == 1).astype(int)
        combined_df['label_name'] = combined_df['label_binary'].map({
            0: 'normal',
            1: 'abnormal'
        })
        
        # Quick summary so I can see the dataset balance
        print(f"\nTraining Dataset Summary:")
        print(f"   Total files: {len(combined_df)}")
        print(f"   Normal (0): {(combined_df['label_binary'] == 0).sum()}")
        print(f"   Abnormal (1): {(combined_df['label_binary'] == 1).sum()}")
        print(f"   Class balance: {(combined_df['label_binary'] == 1).mean():.1%} abnormal")
        
        return combined_df
    
    def load_validation_labels(self) -> Optional[pd.DataFrame]:
        """
        Load validation labels. PhysioNet ships validation separately.
        """
        print("\nLoading validation labels...")
        
        if not self.test_path.exists():
            print("   Warning: Validation directory not found")
            return None
        
        reference_file = self.test_path / 'REFERENCE.csv'
        
        # Same logic as training: skip if there's no label file
        if not reference_file.exists():
            print("   Warning: No REFERENCE.csv found in validation directory")
            return None
        
        df = pd.read_csv(reference_file, names=['filename', 'label'])
        
        # Build full wav file path
        df['wav_path'] = df['filename'].apply(lambda x: str(self.test_path / f"{x}.wav"))
        
        # Filter missing audio files to avoid crashing
        df['file_exists'] = df['wav_path'].apply(lambda p: Path(p).exists())
        
        missing = (~df['file_exists']).sum()
        if missing > 0:
            print(f"   Warning: {missing} files missing in validation")
            df = df[df['file_exists']]
        
        df = df[['filename', 'label', 'wav_path']]
        
        # Convert to binary labels again
        df['label_binary'] = (df['label'] == 1).astype(int)
        df['label_name'] = df['label_binary'].map({0: 'normal', 1: 'abnormal'})
        
        print(f"\nValidation Dataset Summary:")
        print(f"   Total files: {len(df)}")
        print(f"   Normal: {(df['label_binary'] == 0).sum()}")
        print(f"   Abnormal: {(df['label_binary'] == 1).sum()}")
        
        return df
    
    def create_mel_spectrogram_dataset(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the selected audio files into mel spectrograms.
        This is the actual training-ready dataset for CNNs.
        """
        if max_samples:
            df = df.head(max_samples)
        
        X_list = []
        y_list = []
        failed = 0
        
        print(f"\nCreating mel spectrogram dataset...")
        print(f"   Processing {len(df)} files...")
        if augment:
            print("   Augmentation ON")
        
        for idx, row in df.iterrows():
            try:
                # Load raw audio
                waveform = self.preprocessor.load_waveform(Path(row['wav_path']))
                
                # Add augmentation if enabled
                if augment:
                    waveform, _ = self.preprocessor.augment(waveform)
                
                # Convert to mel spectrogram
                mel_spec = self.preprocessor.to_mel_spectrogram(waveform)
                
                X_list.append(mel_spec)
                y_list.append(row['label_binary'])
                
                # Nice little progress update
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(df)} files")
                    
            except Exception as e:
                # If a file is corrupted or unreadable, I just skip it
                failed += 1
                if failed <= 5:
                    print(f"   Warning: Error with {row['filename']}: {e}")
                continue
        
        if failed > 0:
            print(f"   Warning: {failed} files failed to process")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\nDataset created!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Normal: {(y == 0).sum()}, Abnormal: {(y == 1).sum()}")
        
        return X, y
    
    def create_train_val_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the full training dataset into train/val while keeping class balance.
        """
        from sklearn.model_selection import train_test_split
        
        # Get the combined label dataframe
        full_df = self.load_all_training_labels()
        
        # Stratified split keeps the normal/abnormal ratio consistent
        train_df, val_df = train_test_split(
            full_df,
            test_size=test_size,
            random_state=random_state,
            stratify=full_df['label_binary']
        )
        
        print(f"\nTrain/Val Split:")
        print(f"   Training: {len(train_df)} samples")
        print(f"   - Normal: {(train_df['label_binary'] == 0).sum()}")
        print(f"   - Abnormal: {(train_df['label_binary'] == 1).sum()}")
        print(f"   Validation: {len(val_df)} samples")
        print(f"   - Normal: {(val_df['label_binary'] == 0).sum()}")
        print(f"   - Abnormal: {(val_df['label_binary'] == 1).sum()}")
        
        return train_df, val_df
    
    def get_sample_for_prediction(
        self,
        label: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, str, int]:
        """
        Grab one random sample from the dataset.
        Super useful when I'm testing models or visualizing spectrograms.
        """
        df = self.load_all_training_labels()
        
        # If I want only normal/abnormal, I filter
        if label is not None:
            df = df[df['label_binary'] == label]
        
        # Pick a random sample
        sample = df.sample(1).iloc[0]
        
        # Load waveform + convert to mel
        waveform = self.preprocessor.load_waveform(Path(sample['wav_path']))
        mel_spec = self.preprocessor.to_mel_spectrogram(waveform)
        
        return waveform, mel_spec, sample['filename'], sample['label_binary']
