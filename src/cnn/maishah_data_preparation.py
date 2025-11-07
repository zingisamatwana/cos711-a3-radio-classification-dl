import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA PREPARATION PIPELINE ====================

class DataPreparation:
    """
    Complete data preparation pipeline for radio source classification
    """
    
    def __init__(self, typ_dir='typ_PNG/', exo_dir='exo_PNG/', unl_dir='unl_PNG/', 
                 labels_file='labels.csv', test_file='test.csv'):
        self.typ_dir = typ_dir
        self.exo_dir = exo_dir
        self.unl_dir = unl_dir
        self.labels_file = labels_file
        self.test_file = test_file
        
        self.labels_df = None
        self.images_df = None
        self.class_distribution = None
    
    # ==================== STEP 0: EXPLORE IMAGE NAMING ====================
    
    def explore_image_naming(self):
        """
        Explore the actual image naming convention in the directories
        """
        print("=" * 70)
        print("STEP 0: EXPLORING IMAGE NAMING CONVENTION")
        print("=" * 70)
        
        for dir_name, dir_path in [('Typical', self.typ_dir), 
                                    ('Exotic', self.exo_dir), 
                                    ('Unlabeled', self.unl_dir)]:
            print(f"\n {dir_name} Directory: {dir_path}")
            
            if os.path.exists(dir_path): 
                all_files = os.listdir(dir_path)
                image_files = [f for f in all_files 
                              if f.endswith('.png') or f.endswith('.fits.png')]
                
                print(f"   Total files: {len(all_files)}")
                print(f"   Image files: {len(image_files)}")
                 
                print(f"\n   Sample filenames:")
                for i, filename in enumerate(image_files[:5]):
                    print(f"      {i+1}. {filename}")
                 
                if len(image_files) > 0:
                    print(f"\n   Filename analysis:")
                     
                    has_underscore = sum('_' in f for f in image_files[:10])
                    has_fits = sum('.fits.png' in f for f in image_files[:10])
                    
                    print(f"Files with underscores: {has_underscore}/10")
                    print(f"Files with .fits.png: {has_fits}/10")
                     
                    sample_file = image_files[0]
                    print(f"\n   Parsing sample: '{sample_file}'")
                    ra, dec = self.parse_filename_coordinates(sample_file)
                    print(f"      Extracted RA: {ra}")
                    print(f"      Extracted DEC: {dec}")
            else:
                print(f"   WARNING: Directory not found!")
        
        print("\n" + "=" * 70) 
    
    def load_labels(self):
        """
        Load and parse the labels CSV file
        Handles multiple labels per row and coordinate matching
        """
        print("\n" )
        print("STEP 1: LOADING AND PARSING LABELS")
        
        
        df = pd.read_csv(self.labels_file, header=None)
        print(f"\nRaw CSV shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
         
        columns = ['ra', 'dec'] + [f'label_{i}' for i in range(len(df.columns)-2)]
        df.columns = columns
         
        def extract_labels(row):
            labels = []
            for col in df.columns[2:]: 
                val = row[col]
                if pd.notna(val) and str(val).strip() != '':
                    label = str(val).strip()
                    if label not in labels: 
                        labels.append(label)
            return labels
        
        df['labels'] = df.apply(extract_labels, axis=1) 
        df_with_labels = df[df['labels'].apply(len) > 0].copy() 
        self.labels_df = df_with_labels[['ra', 'dec', 'labels']].reset_index(drop=True)
        
        print(f"\nParsed {len(self.labels_df)} labeled sources")
        print(f"Columns: {list(self.labels_df.columns)}") 
        print("\nSample labels:")
        for i in range(min(5, len(self.labels_df))):
            print(f"  Source {i+1} [{self.labels_df.iloc[i]['ra']:.2f}, "
                  f"{self.labels_df.iloc[i]['dec']:.2f}]: {self.labels_df.iloc[i]['labels']}")
        
        return self.labels_df 
    
    def analyze_labels(self):
        """
        Comprehensive label analysis including distribution and statistics
        """
        print("\n")
        print("STEP 2: LABEL ANALYSIS")
        
        
        if self.labels_df is None:
            self.load_labels() 
        all_labels = []
        for labels_list in self.labels_df['labels']:
            all_labels.extend(labels_list)
         
        label_counts = Counter(all_labels)
        
        print(f"\nTotal label instances: {len(all_labels)}")
        print(f"Unique label types: {len(label_counts)}")
        print(f"Sources with labels: {len(self.labels_df)}")
        
        label_lengths = self.labels_df['labels'].apply(len)
        print(f"\nMulti-label Statistics:")
        print(f"Sources with 1 label: {sum(label_lengths == 1)}")
        print(f"Sources with 2 labels: {sum(label_lengths == 2)}")
        print(f"Sources with 3+ labels: {sum(label_lengths >= 3)}")
        print(f"Average labels per source: {label_lengths.mean():.2f}")
        print(f"Max labels on a source: {label_lengths.max()}")
         
        print(f"\nLabel Distribution (sorted by frequency):")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            percentage = (count / len(all_labels)) * 100
            print(f"   • {label:25s}: {count:4d} ({percentage:5.1f}%)")
         
        rare_threshold = len(all_labels) * 0.05
        rare_classes = [label for label, count in label_counts.items() 
                       if count < rare_threshold]
        print(f"\nRare classes (< 5%): {rare_classes}")
         
        self.class_distribution = label_counts
        
        return label_counts
      
    def parse_filename_coordinates(self, filename):
        """
        Extract RA and DEC coordinates from image filename
        Expected format: "RA DEC_[...] deg_(...).fits.png"
        Example: "0.250 -25.084_[0.02238656 0.02238656] deg_(Abell_141_1pln-forPyBDSF.FITS).fits.png"
        """
        try: 
            coord_part = filename.split('_')[0]
             
            coords = coord_part.split()
            
            if len(coords) == 2:
                ra = float(coords[0])
                dec = float(coords[1])
                return ra, dec
        except (ValueError, IndexError) as e:
            pass
        
        return None, None
    
    def find_closest_label(self, source_ra, source_dec, threshold=0.01):
        """
        Find the closest matching label for given coordinates
        Uses Euclidean distance with a threshold
        """
        if source_ra is None or source_dec is None:
            return []
         
        source_coords = np.array([[source_ra, source_dec]])
        label_coords = self.labels_df[['ra', 'dec']].values
         
        distances = cdist(source_coords, label_coords, metric='euclidean')[0]
         
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance < threshold:
            return self.labels_df.iloc[min_idx]['labels']
        
        return [] 
    
    def collect_images_with_labels(self, verbose=True):
        """
        Scan directories and match images with their labels
        """
        print("\n")
        print("STEP 3: COLLECTING AND MATCHING IMAGES")
       
        
        if self.labels_df is None:
            self.load_labels()
        
        image_data = [] 
        if os.path.exists(self.typ_dir): 
            all_files = os.listdir(self.typ_dir)
            typ_files = [f for f in all_files 
                        if (f.endswith('.png') or f.endswith('.fits.png')) 
                        and not f.startswith('.')]
            
            print(f"\nTypical directory: {len(typ_files)} image files")
            
            matched = 0
            unmatched = 0
            for filename in typ_files:
                ra, dec = self.parse_filename_coordinates(filename)
                labels = self.find_closest_label(ra, dec)
                
                if len(labels) > 0:
                    image_data.append({
                        'path': os.path.join(self.typ_dir, filename),
                        'filename': filename,
                        'ra': ra,
                        'dec': dec,
                        'labels': labels,
                        'type': 'typical',
                        'num_labels': len(labels)
                    })
                    matched += 1
                else:
                    unmatched += 1
            
            print(f"   Matched {matched} images with labels")
            print(f"   {unmatched} images without matching labels")
        else:
            print(f"\nWarning: {self.typ_dir} not found")
         
        if os.path.exists(self.exo_dir):
            all_files = os.listdir(self.exo_dir)
            exo_files = [f for f in all_files 
                        if (f.endswith('.png') or f.endswith('.fits.png')) 
                        and not f.startswith('.')]
            
            print(f"\nExotic directory: {len(exo_files)} image files")
            
            matched = 0
            unmatched = 0
            for filename in exo_files:
                ra, dec = self.parse_filename_coordinates(filename)
                labels = self.find_closest_label(ra, dec)
                
                if len(labels) > 0:
                    image_data.append({
                        'path': os.path.join(self.exo_dir, filename),
                        'filename': filename,
                        'ra': ra,
                        'dec': dec,
                        'labels': labels,
                        'type': 'exotic',
                        'num_labels': len(labels)
                    })
                    matched += 1
                else:
                    unmatched += 1
            
            print(f"   Matched {matched} images with labels")
            print(f"   {unmatched} images without matching labels")
        else:
            print(f"\nWarning: {self.exo_dir} not found")
         
        if len(image_data) > 0:
            self.images_df = pd.DataFrame(image_data)
            
            print(f"\nTotal matched images: {len(self.images_df)}")
            print(f"   • Typical: {sum(self.images_df['type'] == 'typical')}")
            print(f"   • Exotic: {sum(self.images_df['type'] == 'exotic')}")
        else:
            print(f"\nNo images matched with labels")
            self.images_df = pd.DataFrame()
        
        return self.images_df 
    
    def check_image_quality(self, sample_size=100):
        """
        Check image quality, sizes, and identify potential issues
        """
        print("\n")
        print("STEP 4: IMAGE QUALITY CHECKS")
        
        
        if self.images_df is None or len(self.images_df) == 0:
            print("\nNo images available for quality check")
            return None 
        sample_df = self.images_df.sample(min(sample_size, len(self.images_df)))
        
        sizes = []
        modes = []
        corrupted = []
        
        print(f"\nChecking {len(sample_df)} sample images...")
        
        for idx, row in sample_df.iterrows():
            try:
                img = Image.open(row['path'])
                sizes.append(img.size)
                modes.append(img.mode)
            except Exception as e:
                corrupted.append(row['path'])
         
        unique_sizes = list(set(sizes))
        print(f"\nImage Dimensions:")
        size_counts = Counter(sizes)
        for size, count in size_counts.most_common(5):
            print(f"{size}: {count} images")
         
        print(f"\nColor Modes:")
        mode_counts = Counter(modes)
        for mode, count in mode_counts.items():
            print(f"   • {mode}: {count} images")
         
        if corrupted:
            print(f"\nCorrupted images found: {len(corrupted)}")
            for path in corrupted[:5]:
                print(f"   • {path}")
        else:
            print(f"\nNo corrupted images detected in sample")
        
        return {
            'sizes': sizes,
            'modes': modes,
            'corrupted': corrupted
        }
      
    def analyze_class_imbalance(self):
        """
        Analyze class imbalance and suggest handling strategies
        """
        print("\n")
        print("STEP 5: CLASS IMBALANCE ANALYSIS")
        
        
        if self.class_distribution is None:
            self.analyze_labels()
         
        counts = np.array(list(self.class_distribution.values()))
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"\nImbalance Statistics:")
        print(f"   • Most common class: {max_count} samples")
        print(f"   • Least common class: {min_count} samples")
        print(f"   • Imbalance ratio: {imbalance_ratio:.2f}:1")
         
        if imbalance_ratio > 100:
            severity = "SEVERE"
            color = "red"
        elif imbalance_ratio > 20:
            severity = "MODERATE"
            color = "yellow"
        else:
            severity = "MILD"
            color = "green"
        
        print(f"\n{color} Imbalance Severity: {severity}")
                
        return imbalance_ratio 
    
    def collect_unlabeled_images(self):
        """
        Collect all unlabeled images for pseudo-labeling
        """
        print("\n")
        print("STEP 6: COLLECTING UNLABELED IMAGES")
        
        
        unlabeled_data = []
        
        if os.path.exists(self.unl_dir):
            all_files = os.listdir(self.unl_dir)
            unl_files = [f for f in all_files 
                        if (f.endswith('.png') or f.endswith('.fits.png')) 
                        and not f.startswith('.')]
            
            print(f"\n Unlabeled directory: {len(unl_files)} image files")
            
            for filename in unl_files:
                ra, dec = self.parse_filename_coordinates(filename)
                
                if ra is not None and dec is not None:
                    unlabeled_data.append({
                        'path': os.path.join(self.unl_dir, filename),
                        'filename': filename,
                        'ra': ra,
                        'dec': dec,
                        'type': 'unlabeled'
                    })
            
            print(f" Collected {len(unlabeled_data)} unlabeled images")
        else:
            print(f"  Warning: {self.unl_dir} not found")
        
        return pd.DataFrame(unlabeled_data)
      
    def prepare_test_set(self):
        """
        Load and prepare test set coordinates
        """
        print("\n")
        print("STEP 7: PREPARING TEST SET")
        
        
        if os.path.exists(self.test_file):
            test_df = pd.read_csv(self.test_file, header=None, names=['ra', 'dec'])
            print(f"\n Loaded {len(test_df)} test coordinates")
            print(f"Sample coordinates:\n{test_df.head()}")
            return test_df
        else:
            print(f"  Warning: {self.test_file} not found")
            return None 
    
    def visualize_data(self, save_path='data_analysis/'):
        """
        Create comprehensive visualizations of the dataset
        """
        print("\n")
        print("STEP 8: DATA VISUALIZATION")
        
        
        os.makedirs(save_path, exist_ok=True)
         
        if self.class_distribution:
            plt.figure(figsize=(12, 6))
            labels = list(self.class_distribution.keys())
            counts = list(self.class_distribution.values())
            
            plt.bar(range(len(labels)), counts, color='steelblue')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Class Distribution', fontsize=14, fontweight='bold')
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'class_distribution.png'), dpi=300)
            print(f" Saved: class_distribution.png")
            plt.close()
         
        if self.images_df is not None and len(self.images_df) > 0:
            plt.figure(figsize=(10, 6))
            label_counts = self.images_df['num_labels'].value_counts().sort_index()
            
            plt.bar(label_counts.index, label_counts.values, color='coral')
            plt.xlabel('Number of Labels per Image', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Multi-Label Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multilabel_distribution.png'), dpi=300)
            print(f" Saved: multilabel_distribution.png")
            plt.close()
         
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            sample_imgs = self.images_df.sample(min(8, len(self.images_df)))
            
            for idx, (_, row) in enumerate(sample_imgs.iterrows()):
                try:
                    img = Image.open(row['path'])
                    axes[idx].imshow(img, cmap='gray')
                    axes[idx].set_title(f"{', '.join(row['labels'][:2])}", 
                                       fontsize=9, wrap=True)
                    axes[idx].axis('off')
                except:
                    axes[idx].text(0.5, 0.5, 'Error loading', 
                                  ha='center', va='center')
                    axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'sample_images.png'), dpi=300)
            print(f" Saved: sample_images.png")
            plt.close()
        
        print(f"\n All visualizations saved to: {save_path}")
      
    def run_complete_preparation(self):
        """
        Run the complete data preparation pipeline
        """
        print("\n" + "=" * 70)
        print("RADIO SOURCE DATA PREPARATION PIPELINE")
        print("=" * 70) 
        self.explore_image_naming() 
        self.load_labels() 
        self.analyze_labels() 
        self.collect_images_with_labels() 
        if self.images_df is not None and len(self.images_df) > 0:
            self.check_image_quality() 
        self.analyze_class_imbalance() 
        unlabeled_df = self.collect_unlabeled_images() 
        test_df = self.prepare_test_set() 
        if self.images_df is not None and len(self.images_df) > 0:
            self.visualize_data()
        
        print("\n" + "=" * 70)
        print(" DATA PREPARATION COMPLETE!")
        print("=" * 70)
        
        print(f"\n Summary:")
        print(f"   • Labeled images: {len(self.images_df) if self.images_df is not None else 0}")
        print(f"   • Unlabeled images: {len(unlabeled_df)}")
        print(f"   • Test coordinates: {len(test_df) if test_df is not None else 0}")
        print(f"   • Unique classes: {len(self.class_distribution) if self.class_distribution else 0}")
        
        return {
            'labeled_images': self.images_df,
            'unlabeled_images': unlabeled_df,
            'test_coordinates': test_df,
            'class_distribution': self.class_distribution
        } 

if __name__ == "__main__": 
    data_prep = DataPreparation(
        typ_dir='typ_PNG/',
        exo_dir='exo_PNG/',
        unl_dir='unl_PNG/',
        labels_file='labels.csv',
        test_file='test.csv'
    ) 
    results = data_prep.run_complete_preparation() 
    if results['labeled_images'] is not None and len(results['labeled_images']) > 0:
        print("\n Saving prepared datasets...")
        results['labeled_images'].to_csv('prepared_labeled_data.csv', index=False)
        results['unlabeled_images'].to_csv('prepared_unlabeled_data.csv', index=False)
        print(" Saved prepared datasets")
    else:
        print("\n WARNING: Images not found")
