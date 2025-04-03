import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

def load_labels(data_path):
    """
    load the label file: labels.csv
    
    Param: Path object, path to the data
    
    Return: Dictionary, keys are IDs (str), values are labels: 1 (benign) or 2 (malignant) or 3 (malignant) 
    """
    df = pd.read_csv(data_path/"labels.csv")
    
    labels = dict(zip(df['ID'], df['Diagnosis']))
    labels = {str(k): v for k, v in labels.items()}
    
    return labels

def extract_fea(scan, mask, n_bins=100):
    """
    Extract radiomic features.
    
    Params:
        scan: 3D array of scan data
        mask: 3D array of segmentation mask
        num_bins: Number of histogram bins
    
    Return: features: Dictionary containing 3 calculated features
    """
    voxels = scan[mask > 0]

    hist, _ = np.histogram(voxels, bins=n_bins, range=(np.min(voxels), np.max(voxels)))
    p = hist / len(voxels)

    energy = np.sum(voxels ** 2)
    mad = np.mean(np.abs(voxels - np.mean(voxels)))
    uniformity =  np.sum(p ** 2)
    
    features = {
        'energy': energy,
        'mad': mad,
        'uniformity': uniformity
    }
    
    return features


def vis_fea(df):
    """
    Visualise radiomics features
    
    Param: df: DataFrame containing features and labels   
    """
    plt.figure(figsize=(15, 10), dpi=300)
    
    feature_columns = [col for col in df.columns if col != 'label']
    
    for i, feature in enumerate(feature_columns):
        plt.subplot(len(feature_columns), 1, i+1)
        sns.boxplot(x='label', y=feature, data=df)
        plt.title(f'{feature} by label')
        plt.tight_layout()
    
    plt.show()
    
    plt.figure(figsize=(10, 8), dpi=300)
    corr = df[feature_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.show()

    g = sns.pairplot(df, hue='label')
    g.fig.set_dpi(300)  
    plt.suptitle('Feature Pairplot by Label', y=1.02)
    plt.show()


def analyse(df):
    """
    Analyse the features
    
    Param: df: DataFrame containing features and labels
    """
    
    fea_cols = [col for col in df.columns if col != 'label']

    results = {}
    
    for feature in fea_cols:
        benign_values = df[df['label'].isin([1])][feature].values         # 1=benign/non-malignant disease
        malignant_values = df[df['label'].isin([2, 3])][feature].values   # 2= malignant, primary lung cancer
                                                                          # 3 = malignant metastatic
        # perform t-test
        t_stat, p_value = stats.ttest_ind(benign_values, malignant_values, equal_var=False)
        
        results[feature] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_benign': np.mean(benign_values),
            'mean_malignant': np.mean(malignant_values),
            'std_benign': np.std(benign_values),
            'std_malignant': np.std(malignant_values)
        }
    
    # convey to DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('p_value')
    
    print("t-test:")
    print(results_df)
    
    return results_df