import pandas as pd
import pyreadstat
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

# Plotting settings for Korean language support
plt.rcParams['font.family'] = 'AppleGothic'  # For macOS
plt.rcParams['axes.unicode_minus'] = False

class HumanRightsDataPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, pyreadstat.metadata_container] = {}

    def load_data(self, file_path: str, name: str):
        """Loads .sav (SPSS) files and stores data and metadata."""
        full_path = os.path.join(self.data_dir, file_path)
        if not os.path.exists(full_path):
            print(f"Error: File not found at {full_path}")
            return

        print(f"Loading {name} from {full_path}...")
        df, meta = pyreadstat.read_sav(full_path)
        self.datasets[name] = df
        self.metadata[name] = meta
        print(f"Successfully loaded {name}. Shape: {df.shape}")

    def get_labeled_df(self, name: str) -> pd.DataFrame:
        """Returns a dataframe with category labels instead of codes."""
        if name not in self.datasets:
            return None
        
        df = self.datasets[name].copy()
        meta = self.metadata[name]
        
        # Apply value labels
        for col, labels in meta.variable_value_labels.items():
            if col in df.columns:
                df[col] = df[col].map(labels)
        
        return df

    def preprocess(self, name: str):
        """Base preprocessing step (handling missing values, etc.)"""
        if name not in self.datasets:
            return
        
        df = self.datasets[name]
        # Example: Fill NaN or drop columns with too many missing values
        # df = df.dropna(subset=['important_column'])
        
        self.datasets[name] = df
        print(f"Preprocessing complete for {name}.")

    def analyze_frequency(self, name: str, column_name: str):
        """Analyzes frequency distribution of a specific column."""
        df_labeled = self.get_labeled_df(name)
        if df_labeled is None or column_name not in df_labeled.columns:
            print(f"Column {column_name} not found in {name}")
            return

        counts = df_labeled[column_name].value_counts()
        print(f"\nFrequency Distribution for {column_name} in {name}:")
        print(counts)
        return counts

    def plot_distribution(self, name: str, column_name: str, title: str = None):
        """Plots frequency distribution."""
        counts = self.analyze_frequency(name, column_name)
        if counts is None:
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(title or f"Distribution of {column_name}")
        plt.xticks(rotation=45)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

def main():
    pipeline = HumanRightsDataPipeline()

    # Define files to load based on the directory structure
    files = {
        "Teacher_HumanRights_2024": "교원 인권상황 실태조사,2024/kor_data_20240073.sav",
        "PreService_Teacher_2023": "교원양성기관 예비교원 인권 및 인권교육에 대한 인식조사,2023/kor_data_20230036.sav",
        "Edu_HumanRights_2021": "초·중등 교원 인권교육 실태조사, 2021/kor_data_20210019.sav",
        "Youth_HumanRights_2024": "(RAWDATA) 2024 아동·청소년 인권실태조사.SAV"
    }

    # Load datasets
    for name, path in files.items():
        pipeline.load_data(path, name)

    # Example: Run EDA on one dataset
    # You would need to check the variable names in the metadata to use specific columns
    # example_name = "Teacher_HumanRights_2024"
    # pipeline.preprocess(example_name)
    
    # print("\nAvailable columns for Teacher_HumanRights_2024:")
    # print(pipeline.metadata[example_name].column_names[:20]) # Show first 20

if __name__ == "__main__":
    main()
