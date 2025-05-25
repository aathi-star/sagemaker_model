import os
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import glob

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_points=2048, target_classes=None):
        """
        ModelNet10 dataset for Halton-MaskGIT 3D generation
        
        Args:
            root_dir: Path to ModelNet10 dataset
            split: 'train' or 'test'
            transform: Optional transforms
            num_points: Number of points to sample from each mesh
            target_classes: Optional list of class names to load (e.g., ['chair', 'table'])
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_points = num_points
        
        # Get all categories
        self.categories = [d for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))]
        self.categories.sort()

        # Determine which categories to load
        load_categories = []
        if target_classes is not None and isinstance(target_classes, list):
            for tc in target_classes:
                if tc in self.categories:
                    load_categories.append(tc)
                else:
                    print(f"Warning: Target class '{tc}' not found in dataset categories. Ignoring.")
            if not load_categories: # If all target classes were invalid, load all as fallback
                print("Warning: No valid target classes specified or found. Loading all classes.")
                load_categories = self.categories
        else:
            load_categories = self.categories # Load all if target_classes is not a list or is None
        
        # Create category to index mapping (only for loaded categories if specified, or all if not)
        # For simplicity in label consistency, we'll keep the original category_to_idx for all dataset categories
        # but only load data from load_categories.
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Get all OFF files
        self.paths = []
        self.labels = []
        
        for category in load_categories: # Iterate over specified or all categories
            category_dir = os.path.join(root_dir, category, split)
            if not os.path.exists(category_dir):
                print(f"Warning: Directory {category_dir} not found for class {category}. Skipping.")
                continue
                
            for file in glob.glob(os.path.join(category_dir, '*.off')):
                self.paths.append(file)
                self.labels.append(self.category_to_idx[category])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        # Load OFF file and convert to point cloud
        try:
            mesh = trimesh.load(path)
            # Sample points from the mesh surface
            points = mesh.sample(self.num_points)
            
            # Center and normalize
            points = points - points.mean(axis=0)
            points = points / np.max(np.abs(points))
            
            # Convert to tensor
            points_tensor = torch.FloatTensor(points)
            
            if self.transform:
                points_tensor = self.transform(points_tensor)
                
            return {
                'points': points_tensor,
                'label': label,
                'category': self.categories[label],
                'path': path
            }
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a placeholder in case of error
            return {
                'points': torch.zeros((self.num_points, 3)),
                'label': label,
                'category': self.categories[label],
                'path': path
            }
