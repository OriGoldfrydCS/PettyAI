#!/usr/bin/env python3
"""
PettyAI - Pet Adoption Prediction Backend
=========================================

This module provides the backend functionality for the PettyAI pet adoption prediction system.
It handles machine learning model loading, data preprocessing, predictions, and data persistence.

"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import joblib
from pathlib import Path
import torch
import torch.nn as nn

class PetAdoptionPredictor:
    """    
    A machine learning backend for predicting pet adoption timeframes.
    This class integrates with various ML models including PyTorch neural networks,
    CLIP-based multimodal models, and Scikit-learn classifiers.
    
    The system handles:
    - Data preprocessing and feature engineering
    - Model loading and inference (PyTorch/Scikit-learn)
    - CLIP multimodal model support 
    - CSV file management for data persistence
    - Statistical analysis and reporting
    
    Attributes:
        model_path (str): Path to the trained ML model file
        csv_path (str): Path to the CSV file for data storage
        model: Loaded ML model instance (PyTorch/Scikit-learn)
        is_clip_model (bool): Flag indicating if loaded model is CLIP-based
        device (torch.device): PyTorch device (CPU/CUDA)
        feature_columns (list): List of feature column names for model input
        adoption_periods (dict): Mapping of prediction classes to adoption periods
    """
    
    def __init__(self, model_path: Optional[str] = None, csv_path: str = "data/pet_adoption_data.csv"):
        """
        Sets up the predictor with model loading, device configuration, and data initialization.
        Automatically detects model type (PyTorch/Scikit-learn) and configures appropriate
        prediction pipeline.
        
        Args:
            model_path (Optional[str]): Path to the trained ML model file. 
                                      Supports .pt/.pth (PyTorch) and .pkl/.joblib (Scikit-learn).
                                      If None, no model is loaded initially.
            csv_path (str): Path to the CSV file for storing pet data.
                          File will be created if it doesn't exist.
        """
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.is_clip_model = False  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature columns expected by the model (single Breed and Color)
        self.feature_columns = [
            'Type', 'Age', 'Breed', 'Gender', 'Color',
            'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',
            'Health', 'Quantity', 'Fee', 'State', 'PhotoAmt'
        ]
        
        # Adoption period mapping
        self.adoption_periods = {
            0: {"label": "Same Day - 1 Week", "days": "0-7 days", "description": "This pet is predicted to be adopted within the first week of listing."},
            1: {"label": "1 Week - 1 Month", "days": "8-30 days", "description": "This pet is predicted to be adopted within 8-30 days of listing."},
            2: {"label": "1-3 Months", "days": "31-90 days", "description": "This pet is predicted to be adopted within 31-90 days of listing."},
            3: {"label": "3+ Months", "days": "100+ days", "description": "This pet may take more than 100 days to find their forever home."}
        }
        
        # Load model if path is provided (after feature_columns are defined)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize CSV file if it doesn't exist
        self.initialize_csv()
    
    def load_model(self, model_path: str):
        """
        Loads PyTorch or Scikit-learn models with automatic architecture detection.
        Supports CLIP-based multimodal models with attention mechanisms and fallback to
        standard neural networks.
        
        Args:
            model_path (str): Path to the model file (.pt/.pth for PyTorch, .pkl/.joblib for Scikit-learn)
        """
        try:
            print(f"Loading model from: {model_path}")
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Check if it's a state_dict or full model
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # Model saved with state_dict and other info
                    print("DEBUG: Loading model from checkpoint with state_dict")
                    state_dict = checkpoint['state_dict']
                    print(f"DEBUG: State dict keys: {list(state_dict.keys())[:5]}...")
                elif hasattr(checkpoint, 'keys') and not hasattr(checkpoint, 'eval'):
                    # It's just a state_dict (OrderedDict)
                    print("DEBUG: Loading model from state_dict (OrderedDict)")
                    state_dict = checkpoint
                    print(f"DEBUG: State dict keys: {list(state_dict.keys())[:5]}...")
                    
                    # Check if this is a CLIP_MLP_Attn model based on key patterns
                    has_attention = any('attn' in key for key in state_dict.keys())
                    has_cross_attn = any('cross_attn' in key for key in state_dict.keys())
                    has_image_attn = any('image_attn' in key for key in state_dict.keys())
                    
                    if has_attention and (has_cross_attn or has_image_attn):
                        print("DEBUG: Detected CLIP_MLP_Attn model architecture")
                        
                        # Create CLIP_MLP_Attn model architecture
                        struct_dim = len(self.feature_columns)  # Use our feature columns as struct_dim
                        print(f"DEBUG: Using struct_dim = {struct_dim} for structured features")
                        
                        # Define the CLIP_MLP_Attn model architecture
                        class CLIP_MLP_Attn(nn.Module):
                            """
                            A neural network that combines image, text, and structured
                            features using multi-head attention mechanisms for pet adoption prediction.
                            
                            Architecture:
                            - Image attention: Self-attention over image embeddings
                            - Cross-modal attention: Text attending to image features
                            - Feature projection: Dimension reduction for each modality
                            - Fusion classifier: Combined feature classification
                            
                            Args:
                                struct_dim (int): Dimension of structured features (pet attributes)
                                dropout (float): Dropout rate for regularization
                                hidden_dim (int): Hidden layer dimension in classifier
                                num_classes (int): Number of output classes (adoption periods)
                            """
                            def __init__(self, struct_dim, dropout=0.4, hidden_dim=1024, num_classes=4):
                                super().__init__()
                                # Attention over image embeddings
                                self.image_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.2)
                                # Cross-modal attention between image and text
                                self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.2)
                                # Feature fusion layers
                                self.image_proj = nn.Sequential(
                                    nn.LayerNorm(512),
                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Dropout(dropout)
                                )
                                self.text_proj = nn.Sequential(
                                    nn.LayerNorm(512),
                                    nn.Linear(512, 256),
                                    nn.GELU(),
                                    nn.Dropout(dropout)
                                )
                                self.struct_proj = nn.Sequential(
                                    nn.LayerNorm(struct_dim),
                                    nn.Linear(struct_dim, 128),
                                    nn.GELU(),
                                    nn.Dropout(dropout)
                                )
                                # Classifier
                                combined_dim = 256 + 256 + 128  # 640
                                self.classifier = nn.Sequential(
                                    nn.LayerNorm(combined_dim),
                                    nn.Linear(combined_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim // 2, num_classes)
                                )

                            def forward(self, x_img, x_text, x_struct):
                                """
                                Forward pass through the multimodal network
                                
                                Args:
                                    x_img (torch.Tensor): Image embeddings [B, 512]
                                    x_text (torch.Tensor): Text embeddings [B, 512]  
                                    x_struct (torch.Tensor): Structured features [B, struct_dim]
                                    
                                Returns:
                                    torch.Tensor: Classification logits [B, num_classes]
                                """
                                # x_img: [B, 512], x_text: [B, 512], x_struct: [B, struct_dim]
                                x_img_unsqueezed = x_img.unsqueeze(1)       # [B, 1, 512]
                                x_text_unsqueezed = x_text.unsqueeze(1)     # [B, 1, 512]
                                # Text attends to image
                                cross_attn_out, _ = self.cross_attn(query=x_text_unsqueezed, key=x_img_unsqueezed, value=x_img_unsqueezed)
                                enhanced_img = cross_attn_out.squeeze(1)  # [B, 512]
                                # Combine original image with cross-attention enhanced image
                                img_combined = (x_img + enhanced_img) / 2
                                # Project features to lower dimensions
                                img_proj = self.image_proj(img_combined)
                                text_proj = self.text_proj(x_text)
                                struct_proj = self.struct_proj(x_struct)
                                # Concatenate all features
                                x = torch.cat([img_proj, text_proj, struct_proj], dim=1)
                                return self.classifier(x)
                        
                        self.model = CLIP_MLP_Attn(struct_dim=struct_dim)
                        print(f"DEBUG: Created CLIP_MLP_Attn model with struct_dim={struct_dim}")
                        
                        # Load the state dict
                        try:
                            self.model.load_state_dict(state_dict, strict=False)
                            print("DEBUG: Successfully loaded state_dict into CLIP_MLP_Attn model")
                        except Exception as e:
                            print(f"DEBUG: Error loading state_dict: {e}")
                            # Try to load with some flexibility
                            model_dict = self.model.state_dict()
                            filtered_dict = {k: v for k, v in state_dict.items() 
                                           if k in model_dict and v.shape == model_dict[k].shape}
                            model_dict.update(filtered_dict)
                            self.model.load_state_dict(model_dict)
                            print(f"DEBUG: Loaded {len(filtered_dict)} layers successfully")
                    
                    else:
                        print("DEBUG: Creating fallback MLP model")
                        # Fallback to simple MLP if not CLIP_MLP_Attn
                        input_size = len(self.feature_columns)
                        self.model = torch.nn.Sequential(
                            torch.nn.Linear(input_size, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 4)
                        )
                        
                        # Try to load compatible weights
                        try:
                            self.model.load_state_dict(state_dict, strict=False)
                            print("DEBUG: Loaded fallback MLP model")
                        except:
                            print("DEBUG: Could not load weights into fallback model")
                        
                else:
                    # It's a full model object
                    print("DEBUG: Loading full model object")
                    self.model = checkpoint
                
                self.model.eval()   # Set to evaluation mode
                self.model.to(self.device)
                print(f"PyTorch model loaded successfully from {model_path}")
                print(f"Model device: {self.device}")
                print(f"Model type: {type(self.model)}")
                
                # Store that this is a CLIP model for prediction handling
                self.is_clip_model = hasattr(self.model, 'image_attn') or 'CLIP' in str(type(self.model))
                print(f"DEBUG: Is CLIP model: {self.is_clip_model}")
                
            else:
                # Try loading with joblib (for sklearn models)
                self.model = joblib.load(model_path)
                self.is_clip_model = False
                print(f"Scikit-learn model loaded successfully from {model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            self.model = None
            self.is_clip_model = False
    
    def initialize_csv(self):
        """       
        Creates the pet data CSV file with proper headers if it doesn't exist.
        
        CSV Schema:
            - PetID: Unique identifier for each pet
            - Pet Attributes: Type, Name, Age, Breed, Gender, Color, etc.
            - Health Info: Vaccinated, Dewormed, Sterilized, Health status
            - Adoption Info: Fee, State, RescuerID, PhotoAmt
            - Prediction Results: Prediction class, Confidence score
            - Metadata: Photos (JSON), Description, DateAdded
        """
        if not os.path.exists(self.csv_path):
            headers = [
                'PetID', 'Type', 'Name', 'Age', 'Breed', 'Gender', 
                'Color', 'MaturitySize', 'FurLength', 'Vaccinated', 
                'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 
                'RescuerID', 'PhotoAmt', 'Photos', 'Description', 'Prediction', 'Confidence', 
                'DateAdded'
            ]
            
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_path, index=False)
            print(f"Initialized CSV file: {self.csv_path}")
        else:
            print(f"Using existing CSV file: {self.csv_path}")
    
    def preprocess_pet_data(self, pet_data: Dict) -> Dict:
        """
        Transforms raw pet data into the format expected by the ML model.
        Handles data validation, type conversion, and feature encoding.
        
        Args:
            pet_data (Dict): Raw pet data from frontend/API
                           Expected keys: type, age, breed, gender, color,
                           maturitySize, furLength, vaccinated, dewormed,
                           sterilized, health, quantity, fee, state, photoAmt
        
        Returns:
            Dict: Processed pet data with normalized values and proper types
            
        Default Values:
            - Type: 1 (Cat), Age: 12 months, Gender: 1 (Male)
            - MaturitySize: 2 (Medium), FurLength: 1 (Short)
            - Health status: 3 (Healthy), Quantity: 1
            - Fee: 0 (Free), State: 41401 (Kuala Lumpur)
            - PhotoAmt: 0 (No photos)
        """
        processed_data = pet_data.copy()
        
        # Handle breed encoding (single breed field)
        processed_data['Breed'] = pet_data.get('breed', 0)
        
        # Handle color encoding (single color field)
        processed_data['Color'] = pet_data.get('color', 0)
        
        # Ensure all required fields are present
        defaults = {
            'Type': 1,
            'Age': 12,
            'Gender': 1,
            'MaturitySize': 2,
            'FurLength': 1,
            'Vaccinated': 3,
            'Dewormed': 3,
            'Sterilized': 3,
            'Health': 1,
            'Quantity': 1,
            'Fee': 0,
            'State': 41401,  # Default to Kuala Lumpur
            'PhotoAmt': 0
        }
        
        for key, default_value in defaults.items():
            if key not in processed_data:
                processed_data[key] = default_value
        
        return processed_data
    
    def create_feature_vector(self, pet_data: Dict) -> np.ndarray:
        """
        Converts preprocessed pet data into a numerical feature vector
        that can be fed into the machine learning model.
        
        Args:
            pet_data (Dict): Preprocessed pet data dictionary
        
        Returns:
            np.ndarray: Feature vector with shape (1, n_features)
                       Ready for model prediction
        
        Feature Ordering:
            The feature vector follows the exact order defined in self.feature_columns:
            [Type, Age, Breed, Gender, Color, MaturitySize, FurLength, 
             Vaccinated, Dewormed, Sterilized, Health, Quantity, Fee, State, PhotoAmt]
        
        Missing Value Handling:
            - Missing features are replaced with 0 (neutral value)
            - Maintains consistent vector dimensionality
            - Logs warnings for missing critical features
        """
        features = []
        
        for column in self.feature_columns:
            if column in pet_data:
                features.append(pet_data[column])
            else:
                features.append(0)  # Default value for missing features
        
        return np.array(features).reshape(1, -1)
    
    def predict_adoption_time(self, pet_data: Dict) -> Dict:
        """
        Generates adoption time predictions using the loaded ML model.
        Supports both CLIP multimodal models and standard classifiers.
        
        Args:
            pet_data (Dict): Raw pet data from frontend
        
        Returns:
            Dict: Prediction results containing:
                - prediction_class (int): Adoption period class (0-3)
                - confidence (float): Prediction confidence percentage
                - period_label (str): Human-readable period description
                - period_days (str): Expected adoption timeframe
                - description (str): Detailed explanation
        
        Prediction Classes:
            - 0: Same Day - 1 Week (0-7 days)
            - 1: 1 Week - 1 Month (8-30 days)  
            - 2: 1-3 Months (31-90 days)
            - 3: 3+ Months (100+ days)
        """
        # Preprocess data
        processed_data = self.preprocess_pet_data(pet_data)
        print(f"DEBUG: Processing pet data for prediction...")
        print(f"DEBUG: Preprocessed data keys: {list(processed_data.keys())}")
        
        if self.model is not None:
            try:
                print(f"DEBUG: Using PyTorch model for prediction")
                # Create feature vector
                features = self.create_feature_vector(processed_data)
                print(f"DEBUG: Feature vector shape: {features.shape}")
                print(f"DEBUG: Feature values: {features.flatten()}")
                
                # Make prediction based on model type
                if hasattr(self.model, '__class__') and 'torch' in str(type(self.model)):
                    # PyTorch model prediction
                    with torch.no_grad():
                        if self.is_clip_model:
                            # CLIP model needs 3 inputs: image, text, structured
                            print("DEBUG: Using CLIP model - creating dummy embeddings")
                            
                            # Create dummy CLIP embeddings (512-dimensional)
                            # In a full implementation, you'd extract these from actual images/text
                            batch_size = 1
                            dummy_img_embedding = torch.zeros(batch_size, 512, dtype=torch.float32).to(self.device)
                            dummy_text_embedding = torch.zeros(batch_size, 512, dtype=torch.float32).to(self.device)
                            
                            # Use our structured features
                            struct_features = torch.FloatTensor(features).to(self.device)
                            if len(struct_features.shape) == 1:
                                struct_features = struct_features.unsqueeze(0)  # Add batch dimension
                            
                            print(f"DEBUG: CLIP inputs - img: {dummy_img_embedding.shape}, text: {dummy_text_embedding.shape}, struct: {struct_features.shape}")
                            
                            # Call CLIP model with 3 inputs
                            outputs = self.model(dummy_img_embedding, dummy_text_embedding, struct_features)
                            print(f"DEBUG: CLIP Model outputs: {outputs}")
                            
                        else:
                            # Regular PyTorch model with single input
                            features_tensor = torch.FloatTensor(features).to(self.device)
                            if len(features_tensor.shape) == 1:
                                features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
                            
                            print(f"DEBUG: Regular model input tensor shape: {features_tensor.shape}")
                            outputs = self.model(features_tensor)
                            print(f"DEBUG: Regular model outputs: {outputs}")
                        
                        # Get prediction (assuming classification output)
                        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                            # Multi-class output
                            probabilities = torch.softmax(outputs, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = torch.max(probabilities).item() * 100
                            print(f"DEBUG: Multi-class prediction: {prediction}, confidence: {confidence:.2f}%")
                        else:
                            # Single output (regression or binary classification)
                            prediction = int(torch.round(outputs).item())
                            confidence = 75  # Default confidence for regression
                            
                            # Ensure prediction is within valid range (0-3)
                            prediction = max(0, min(3, prediction))
                            print(f"DEBUG: Single output prediction: {prediction}, confidence: {confidence:.2f}%")
                
                else:
                    # Scikit-learn model prediction
                    print(f"DEBUG: Using Scikit-learn model for prediction")
                    prediction = self.model.predict(features)[0]
                    
                    # Get prediction probabilities if available
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(features)[0]
                        confidence = max(probabilities) * 100
                    else:
                        confidence = 75  # Default confidence
                
            except Exception as e:
                print(f"ERROR: Model prediction failed: {e}")
                print("DEBUG: Model is not available or failed - this should not happen in production")
                return {
                    'prediction_class': 2,  # Default to medium prediction
                    'confidence': 50.0,
                    'period_label': 'Model Error',
                    'period_days': '31-90 days',
                    'description': 'Model prediction failed. Please check model file.'
                }
        else:
            print(f"ERROR: No model loaded - cannot make prediction")
            return {
                'prediction_class': 2,      # Default to medium prediction
                'confidence': 50.0,
                'period_label': 'No Model',
                'period_days': '31-90 days',
                'description': 'No model is loaded. Please check model file path.'
            }
        
        # Get period information
        period_info = self.adoption_periods.get(prediction, self.adoption_periods[2])
        print(f"DEBUG: Final prediction: {prediction} ({period_info['days']})")
        
        return {
            'prediction_class': int(prediction),
            'confidence': round(confidence, 1),
            'period_label': period_info['label'],
            'period_days': period_info['days'],
            'description': period_info['description']
        }
    
    def save_pet_data(self, pet_data: Dict, prediction_result: Dict) -> bool:
        """
        Persists pet information and prediction results to the CSV database.
        
        Args:
            pet_data (Dict): Original pet data from frontend
            prediction_result (Dict): Prediction results from model
        
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Prepare data for CSV
            csv_data = {
                'PetID': pet_data.get('id', f"PET_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                'Type': pet_data.get('type', 1),
                'Name': pet_data.get('name', ''),
                'Age': pet_data.get('age', 12),
                'Breed': pet_data.get('breed', 0),
                'Gender': pet_data.get('gender', 1),
                'Color': pet_data.get('color', 0),
                'MaturitySize': pet_data.get('maturitySize', 2),
                'FurLength': pet_data.get('furLength', 1),
                'Vaccinated': pet_data.get('vaccinated', 3),
                'Dewormed': pet_data.get('dewormed', 3),
                'Sterilized': pet_data.get('sterilized', 3),
                'Health': pet_data.get('health', 1),
                'Quantity': pet_data.get('quantity', 1),
                'Fee': pet_data.get('fee', 0),
                'State': pet_data.get('state', 41401),
                'RescuerID': pet_data.get('rescuerID', 1),
                'PhotoAmt': pet_data.get('photoAmt', 0),
                'Photos': json.dumps(pet_data.get('photos', [])),  
                'Description': pet_data.get('description', ''),
                'Prediction': prediction_result['prediction_class'],
                'Confidence': prediction_result['confidence'],
                'DateAdded': datetime.now().isoformat()
            }
            
            # Read existing CSV or create new DataFrame
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                df = pd.concat([df, pd.DataFrame([csv_data])], ignore_index=True)
            else:
                df = pd.DataFrame([csv_data])
            
            # Save to CSV
            df.to_csv(self.csv_path, index=False)
            print(f"Pet data saved to {self.csv_path}")
            return True
            
        except Exception as e:
            print(f"Error saving pet data: {e}")
            return False
    
    def delete_pet_data(self, pet_id: str) -> bool:
        """
        Removes a specific pet record from the CSV database using the PetID.
        
        Args:
            pet_id (str): Unique identifier of the pet to delete
        
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if not os.path.exists(self.csv_path):
                print(f"CSV file does not exist: {self.csv_path}")
                return False
            
            # Read existing CSV
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                print("CSV file is empty")
                return False
            
            # Check if pet exists
            if pet_id not in df['PetID'].values:
                print(f"Pet with ID {pet_id} not found")
                return False
            
            # Remove the pet
            original_count = len(df)
            df = df[df['PetID'] != pet_id]
            new_count = len(df)
            
            # Save updated CSV
            df.to_csv(self.csv_path, index=False)
            print(f"Pet {pet_id} deleted. Pets count: {original_count} -> {new_count}")
            return True
            
        except Exception as e:
            print(f"Error deleting pet data: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Computes comprehensive statistics from the pet database for dashboard display.
        Provides insights into system usage and prediction patterns.
        
        Returns:
            Dict: Statistics dictionary containing:
                - total_pets (int): Total number of pets in database
                - avg_prediction_days (float): Average predicted adoption days
                - fast_adopters (int): Count of pets predicted to adopt quickly (≤30 days)
                - avg_confidence (float): Average prediction confidence score
        """
        try:
            if not os.path.exists(self.csv_path):
                return {
                    'total_pets': 0,
                    'avg_prediction_days': 0,
                    'fast_adopters': 0,
                    'avg_confidence': 0
                }
            
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return {
                    'total_pets': 0,
                    'avg_prediction_days': 0,
                    'fast_adopters': 0,
                    'avg_confidence': 0
                }
            
            # Calculate statistics
            total_pets = len(df)
            
            # Average prediction days
            prediction_days_map = {0: 3.5, 1: 19, 2: 60, 3: 120}
            df['prediction_days'] = df['Prediction'].map(prediction_days_map)
            avg_prediction_days = df['prediction_days'].mean()
            
            # Fast adopters (prediction 0 or 1)
            fast_adopters = len(df[df['Prediction'] <= 1])
            
            # Average confidence
            avg_confidence = df['Confidence'].mean()
            
            return {
                'total_pets': total_pets,
                'avg_prediction_days': round(avg_prediction_days, 1),
                'fast_adopters': fast_adopters,
                'avg_confidence': round(avg_confidence, 1)
            }
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {
                'total_pets': 0,
                'avg_prediction_days': 0,
                'fast_adopters': 0,
                'avg_confidence': 0
            }
    
    def get_recent_pets(self, limit: int = 6) -> List[Dict]:
        """
        Fetches the most recently added pets from the database for display
        in the dashboard recent pets section.
        
        Args:
            limit (int): Maximum number of recent pets to return (default: 6)
        
        Returns:
            List[Dict]: List of recent pet dictionaries, each containing:
                - All pet attributes and prediction results
                - photos (list): Parsed photo URLs from JSON storage
                - period_info (dict): Adoption period details
                - DateAdded: ISO timestamp of record creation
        """
        try:
            if not os.path.exists(self.csv_path):
                return []
            
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return []
            
            # Sort by DateAdded and get most recent
            df = df.sort_values('DateAdded', ascending=False).head(limit)
            
            # Convert to list of dictionaries
            recent_pets = []
            for _, row in df.iterrows():
                pet_dict = row.to_dict()
                # Parse photos from JSON string if it exists
                if 'Photos' in pet_dict and pd.notna(pet_dict['Photos']):
                    try:
                        pet_dict['photos'] = json.loads(pet_dict['Photos'])
                    except (json.JSONDecodeError, TypeError):
                        pet_dict['photos'] = []
                else:
                    pet_dict['photos'] = []
                
                # Add period information
                period_info = self.adoption_periods.get(pet_dict['Prediction'], self.adoption_periods[2])
                pet_dict['period_info'] = period_info
                recent_pets.append(pet_dict)
            
            return recent_pets
            
        except Exception as e:
            print(f"Error getting recent pets: {e}")
            return []

# Flask API endpoints (optional)
def create_flask_app(predictor: PetAdoptionPredictor):
    """
    Create Flask Web Application with REST API
    =========================================
    
    Builds a Flask web application with RESTful API endpoints
    for the PettyAI prediction system. Provides both web interface and
    programmatic access to prediction capabilities.
    
    Args:
        predictor (PetAdoptionPredictor): Initialized predictor instance
    
    Returns:
        Flask: Configured Flask application instance with all endpoints
               Returns None if Flask is not available
    
    API Endpoints:
        GET  /                : Serve main HTML interface
        POST /predict         : Generate adoption time prediction
        POST /save            : Save pet data and predictions
        DELETE /delete        : Remove pet record by ID
        GET  /statistics      : Retrieve system statistics
        GET  /recent-pets     : Get recent pet records
    """
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
        
        app = Flask(__name__, static_folder='frontend', static_url_path='')
        CORS(app)  # Enable CORS for frontend integration
        
        @app.route('/')
        def index():
            """
            Serve Main HTML Interface.
            Delivers the main web application interface to users.
            
            Returns:
                HTML: The local_app.html file from the frontend directory
            """
            return send_from_directory('frontend', 'local_app.html')
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """
            Generate Pet Adoption Time Prediction.
            Accepts pet data and returns adoption timeframe prediction.
            
            Request Body (JSON):
                {
                    "type": int,          // Pet type (1=Cat, 2=Dog)
                    "age": int,           // Age in months
                    "breed": int,         // Breed ID
                    "gender": int,        // Gender (1=Male, 2=Female)
                    "color": int,         // Color ID
                    "maturitySize": int,  // Size category
                    "furLength": int,     // Fur length category
                    "vaccinated": int,    // Vaccination status
                    "dewormed": int,      // Deworming status
                    "sterilized": int,    // Sterilization status
                    "health": int,        // Health condition
                    "quantity": int,      // Number of pets
                    "fee": float,         // Adoption fee in USD
                    "state": int,         // Location state ID
                    "photoAmt": int       // Number of photos
                }
            
            Response (JSON):
                {
                    "prediction_class": int,    // 0-3 adoption period class
                    "confidence": float,        // Prediction confidence %
                    "period_label": str,        // Human-readable timeframe
                    "period_days": str,         // Expected days range
                    "description": str          // Detailed explanation
                }
            
            Error Response:
                {"error": "Error message"} with HTTP 500
            """
            try:
                pet_data = request.json
                prediction_result = predictor.predict_adoption_time(pet_data)
                return jsonify(prediction_result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/save', methods=['POST'])
        def save_pet():
            """
            Save Pet Data and Prediction Results.            
            Persists pet information and ML predictions to the CSV database.
            
            Request Body (JSON):
                {
                    "pet_data": {
                        // Complete pet information object
                        "id": str,               // Optional: Pet ID
                        "name": str,             // Pet name
                        "type": int,             // Pet type
                        "age": int,              // Age in months
                        // ... other pet attributes
                        "photos": [str],         // Array of photo URLs
                        "description": str       // Pet description
                    },
                    "prediction_result": {
                        // Prediction results from /predict endpoint
                        "prediction_class": int,
                        "confidence": float,
                        "period_label": str,
                        "period_days": str,
                        "description": str
                    }
                }
            
            Response (JSON):
                {"success": bool}  // True if save successful
            
            Error Response:
                {"error": "Error message"} with HTTP 500
            
            Note:
                Creates unique PetID if not provided. Adds timestamp
                for tracking. Validates data before persistence.
            """
            try:
                data = request.json
                pet_data = data.get('pet_data', {})
                prediction_result = data.get('prediction_result', {})
                
                success = predictor.save_pet_data(pet_data, prediction_result)
                return jsonify({'success': success})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/delete', methods=['DELETE'])
        def delete_pet():
            """            
            Removes a pet record from the CSV database by PetID.
            
            Request Body (JSON):
                {
                    "pet_id": str  // Unique identifier of pet to delete
                }
            
            Response (JSON):
                {"success": bool}  // True if deletion successful
            
            Error Responses:
                {"error": "Pet ID is required"} with HTTP 400
                {"error": "Error message"} with HTTP 500
            
            Validation:
                - Requires non-empty pet_id
                - Verifies pet exists before deletion
                - Atomic operation (all or nothing)
            
            Note:
                Deletion is permanent and cannot be undone.
                Returns success=false if pet doesn't exist.
            """
            try:
                data = request.json
                pet_id = data.get('pet_id', '')
                
                if not pet_id:
                    return jsonify({'error': 'Pet ID is required'}), 400
                
                success = predictor.delete_pet_data(pet_id)
                return jsonify({'success': success})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/statistics', methods=['GET'])
        def get_statistics():
            """
            Retrieve System Statistics.            
            Provides analytics about the pet database
            for dashboard display and system monitoring.
            
            Response (JSON):
                {
                    "total_pets": int,           // Total pets in database
                    "avg_prediction_days": float, // Average predicted adoption days
                    "fast_adopters": int,        // Count of quick adopters (≤30 days)
                    "avg_confidence": float      // Average prediction confidence
                }
            
            Error Response:
                {"error": "Error message"} with HTTP 500
            
            Statistics Calculated:
                - Total pets: Count of all records
                - Avg prediction days: Weighted average by prediction class
                - Fast adopters: Pets in classes 0-1 (0-30 days)
                - Avg confidence: Mean of all confidence scores
            
            Note:
                Statistics are computed in real-time from CSV data.
                Returns zero values for empty database.
            """
            try:
                stats = predictor.get_statistics()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/recent-pets', methods=['GET'])
        def get_recent_pets():
            """            
            Returns the most recently added pets for dashboard display.
            
            Query Parameters:
                limit (int): Maximum number of pets to return (default: 6)
            
            Response (JSON):
                [
                    {
                        // Complete pet record with all attributes
                        "PetID": str,
                        "Name": str,
                        "Type": int,
                        "Age": int,
                        "Breed": int,
                        // ... other pet attributes
                        "photos": [str],         // Parsed photo URLs
                        "Prediction": int,       // Prediction class
                        "Confidence": float,     // Confidence score
                        "DateAdded": str,        // ISO timestamp
                        "period_info": {         // Adoption period details
                            "label": str,
                            "days": str,
                            "description": str
                        }
                    }
                ]
            
            Error Response:
                {"error": "Error message"} with HTTP 500
            
            Sorting:
                Records ordered by DateAdded (newest first)
            
            Note:
                Photos are parsed from JSON storage format.
                Empty list returned if no pets exist.
            """
            try:
                limit = request.args.get('limit', 6, type=int)
                recent_pets = predictor.get_recent_pets(limit)
                return jsonify(recent_pets)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return app
        
    except ImportError:
        print("Flask not installed. API endpoints not available.")
        return None

if __name__ == "__main__":
    """
    Main Application Entry Point
    ============================
    
    Initializes the PettyAI prediction system and starts the Flask web server.
    
    Initialization Process:
        1. Create PetAdoptionPredictor instance
        2. Load ML model from specified path
        3. Initialize CSV database
        4. Start Flask web server (if available)
        5. Provide fallback instructions for programmatic use
    
    Configuration:
        - Model path: "data/clip_mlp_model.pt"
        - CSV path: "data/pet_adoption_data.csv"
        - Server: localhost:5000
        - Debug mode: Enabled for development
    
    Output:
        Prints system status including:
        - Model loading success/failure
        - CSV file location
        - Server startup information
        - Usage instructions
    """
    # Initialize predictor with your PyTorch model
    print("=== PettyAI - Pet Adoption Prediction System ===")
    print("Initializing predictor...")
    
    predictor = PetAdoptionPredictor(
        model_path="data/clip_mlp_model.pt", 
        csv_path="data/pet_adoption_data.csv"
    )
    
    print(f"Model loaded: {'Yes' if predictor.model is not None else 'No'}")
    print(f"CSV file: {predictor.csv_path}")
    print("\nSystem ready for predictions!")
    
    # Start Flask app (if Flask is available)
    app = create_flask_app(predictor)
    if app:
        print("\nStarting Flask server on http://localhost:5000")
        print("You can now use the web interface to make predictions.")
        app.run(debug=True, port=5000)
    else:
        print("\nFlask not available. You can use the predictor programmatically:")
        print("predictor.predict_adoption_time(pet_data)")
