import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, classification_report, average_precision_score,
                            roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

class BankChurnPredictor:
    """
    Advanced bank customer churn prediction with enhanced feature engineering,
    optimal model selection, and comprehensive evaluation.
    """
    
    def __init__(self, file_path=None, output_dir='churn_model_outputs'):
        """
        Initialize the churn predictor with file path and output directory
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None
        self.df_engineered = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None
        self.best_threshold = 0.5
        self.preprocessor = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self, file_path=None):
        """
        Load the bank customer data from CSV
        """
        if file_path:
            self.file_path = file_path
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Successfully loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except FileNotFoundError:
            print(f"File not found at {self.file_path}")
            # Try alternative paths
            try:
                alt_paths = [
                    os.path.join(os.getcwd(), 'Bank Customer Churn Prediction.csv'),
                    os.path.join(os.path.dirname(os.getcwd()), 'Bank Customer Churn Prediction.csv')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.df = pd.read_csv(alt_path)
                        print(f"Found file at alternate location: {alt_path}")
                        break
                else:
                    raise FileNotFoundError("Could not find the data file in any location")
            except Exception as e:
                print(f"Error loading data: {e}")
                raise
        
        return self.df
    
    def explore_data(self):
        """
        Perform initial exploratory data analysis
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Basic data overview
        print("\nFirst few rows of the dataset:")
        print(self.df.head())
        
        print("\nData types and non-null values:")
        print(self.df.info())
        
        print("\nSummary statistics:")
        print(self.df.describe())
        
        # Class distribution
        print("\nChurn Distribution:")
        churn_counts = self.df['churn'].value_counts()
        churn_percents = self.df['churn'].value_counts(normalize=True).map(lambda x: f"{x:.2%}")
        
        for i, (count, percent) in enumerate(zip(churn_counts, churn_percents)):
            print(f"Class {i}: {count} customers ({percent})")
        
        # Create visualizations for the class distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='churn', data=self.df, palette='viridis')
        plt.title('Class Distribution: Churn vs Non-Churn')
        plt.xlabel('Churn (1 = Yes, 0 = No)')
        plt.ylabel('Number of Customers')
        
        # Add percentage annotations
        total = len(self.df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')
        
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'))
        plt.close()
        
        # Correlation heatmap of numerical features
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        plt.figure(figsize=(12, 10))
        correlation = self.df[numerical_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # Boxplots of key features vs churn
        key_features = ['age', 'balance', 'credit_score', 'tenure', 'estimated_salary']
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(key_features):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='churn', y=feature, data=self.df)
            plt.title(f'{feature.capitalize()} vs Churn')
            plt.xlabel('Churn')
            plt.ylabel(feature.capitalize())
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_vs_churn_boxplots.png'))
        plt.close()
        
        return self.df
    
    def add_engineered_features(self, df=None):
        """
        Create advanced features for better model performance
        """
        if df is None:
            df = self.df.copy()
        
        # Financial features
        df['balance_to_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
        df['credit_score_to_age_ratio'] = df['credit_score'] / (df['age'] + 1)
        df['zero_balance'] = (df['balance'] == 0).astype(int)
        df['high_balance'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)
        
        # Customer engagement features
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 2, 5, 8, 10], labels=['New', 'Developing', 'Established', 'Loyal'])
        df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        df['credit_level'] = pd.cut(df['credit_score'], bins=[300, 500, 650, 750, 900], labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Customer activity and product features
        df['products_per_tenure'] = df['products_number'] / (df['tenure'] + 1)
        df['is_inactive'] = ((df['balance'] == 0) & (df['active_member'] == 0)).astype(int)
        df['engagement_score'] = df['active_member'] * df['tenure'] * df['products_number']
        df['has_credit_card_inactive'] = ((df['credit_card'] == 1) & (df['active_member'] == 0)).astype(int)
        df['age_to_tenure_ratio'] = df['age'] / (df['tenure'] + 1)
        df['balance_per_product'] = df['balance'] / (df['products_number'] + 0.1)
        
        # Customer lifecycle stage proxy
        df['lifecycle_stage'] = pd.cut(
            df['age'] * df['tenure'], 
            bins=[0, 100, 300, 600, 1000], 
            labels=['Acquisition', 'Growth', 'Maturity', 'Decline']
        )
        
        # Country-specific relative features
        for country in df['country'].unique():
            country_mask = df['country'] == country
            
            # Only calculate if we have data for this country
            if country_mask.sum() > 0:
                country_avg_balance = df.loc[country_mask, 'balance'].mean()
                country_avg_salary = df.loc[country_mask, 'estimated_salary'].mean()
                
                # Customer's balance and salary relative to country average
                if country_avg_balance > 0:
                    df.loc[country_mask, 'balance_to_country_avg'] = df.loc[country_mask, 'balance'] / (country_avg_balance + 1)
                
                if country_avg_salary > 0:
                    df.loc[country_mask, 'salary_to_country_avg'] = df.loc[country_mask, 'estimated_salary'] / (country_avg_salary + 1)
        
        # Advanced interaction features
        df['high_balance_no_activity'] = ((df['balance'] > df['balance'].quantile(0.75)) & 
                                         (df['active_member'] == 0)).astype(int)
        
        df['many_products_low_activity'] = ((df['products_number'] >= 3) & 
                                            (df['active_member'] == 0)).astype(int)
        
        df['young_inactive'] = ((df['age'] < 30) & (df['active_member'] == 0)).astype(int)
        df['elder_inactive'] = ((df['age'] > 60) & (df['active_member'] == 0)).astype(int)
        
        # Credit utilization proxy
        df['credit_utilization'] = df['balance'] / (df['credit_score'] + 1) * df['credit_card']
        
        # RFM-like proxies (Recency, Frequency, Monetary)
        # Using available data as proxies for these traditional RFM metrics
        df['active_tenure_ratio'] = df['active_member'] * df['tenure'] / 10  # 10 max tenure
        
        # Z-score features to identify outliers
        numeric_cols = ['age', 'credit_score', 'balance', 'estimated_salary', 'tenure']
        for col in numeric_cols:
            df[f'{col}_zscore'] = stats.zscore(df[col], nan_policy='omit')
        
        # Volatility indicators (if we assume extreme values indicate volatility)
        df['financial_volatility'] = ((df['balance_zscore'].abs() > 2) | 
                                     (df['estimated_salary_zscore'].abs() > 2)).astype(int)
        
        # Age-based generational cohorts
        df['generation'] = pd.cut(
            df['age'],
            bins=[18, 25, 40, 55, 75, 100],
            labels=['Gen Z', 'Millennials', 'Gen X', 'Boomers', 'Silent']
        )
        
        # Interaction terms between key features
        df['credit_balance_interaction'] = df['credit_score'] * df['balance'] / 10000
        df['tenure_active_interaction'] = df['tenure'] * df['active_member']
        df['age_credit_interaction'] = df['age'] * df['credit_score'] / 100
        
        # High-risk combinations
        df['high_risk_segment'] = ((df['is_inactive'] == 1) & 
                                  (df['age'] < 40) & 
                                  (df['tenure'] < 3)).astype(int)
        
        self.df_engineered = df
        return df
    
    def handle_outliers(self, df=None, columns=None, method='clip', threshold=3):
        """
        Handle outliers in the specified columns
        """
        if df is None:
            df = self.df_engineered.copy() if self.df_engineered is not None else self.df.copy()
        
        if columns is None:
            columns = ['balance', 'estimated_salary', 'credit_score']
        
        df_processed = df.copy()
        
        if method == 'clip':
            for col in columns:
                if col in df_processed.columns:
                    q1 = df_processed[col].quantile(0.25)
                    q3 = df_processed[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (threshold * iqr)
                    upper_bound = q3 + (threshold * iqr)
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                    
        elif method == 'remove':
            for col in columns:
                if col in df_processed.columns:
                    z_scores = stats.zscore(df_processed[col], nan_policy='omit')
                    abs_z_scores = np.abs(z_scores)
                    outlier_mask = abs_z_scores > threshold
                    print(f"Removing {outlier_mask.sum()} outliers from {col}")
                    df_processed = df_processed[~outlier_mask]
        
        return df_processed
    
    def prepare_data(self, test_size=0.2, val_size=0.25, stratify=True, handle_outliers=True):
        """
        Prepare data for modeling by:
        1. Dropping unnecessary columns
        2. Handling outliers
        3. Splitting into train/val/test sets
        """
        if self.df_engineered is None:
            self.df_engineered = self.add_engineered_features(self.df)
        
        # Drop customer_id column as it's not useful for modeling
        if 'customer_id' in self.df_engineered.columns:
            self.df_engineered = self.df_engineered.drop(['customer_id'], axis=1)
        
        # Handle outliers if requested
        if handle_outliers:
            numeric_cols = self.df_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'churn']  # Exclude target
            df_processed = self.handle_outliers(self.df_engineered, numeric_cols, method='clip')
        else:
            df_processed = self.df_engineered
        
        # Split features and target variable
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        
        # Create train, validation, and test sets with stratification
        if stratify:
            # First split: training and temp (validation + test)
            temp_size = test_size + val_size * (1 - test_size)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=temp_size, random_state=42, stratify=y
            )
            
            # Second split: validation and test from temp
            test_ratio = test_size / temp_size
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_ratio, random_state=42, stratify=y_temp
            )
        else:
            # Non-stratified split (not recommended for imbalanced data)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=temp_size, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_ratio, random_state=42
            )
        
        # Store the splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing transformers
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='passthrough'
        )
        
        print(f"\nData Split:")
        print(f"Training set size: {X_train.shape[0]} samples ({y_train.mean()*100:.1f}% churn)")
        print(f"Validation set size: {X_val.shape[0]} samples ({y_val.mean()*100:.1f}% churn)")
        print(f"Test set size: {X_test.shape[0]} samples ({y_test.mean()*100:.1f}% churn)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote(self, X_train=None, y_train=None, sampling_strategy=0.8):
        """
        Apply SMOTE to handle class imbalance
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        # Transform the data first
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
        
        return X_train_resampled, y_train_resampled
    
    def build_models(self, use_smote=True, sampling_strategy=0.8):
        """
        Build and configure multiple models for the churn prediction task
        """
        # Apply preprocessing to the training data
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        X_val_transformed = self.preprocessor.transform(self.X_val)
        
        # Apply SMOTE if requested
        if use_smote:
            X_train_resampled, y_train_resampled = self.apply_smote(
                sampling_strategy=sampling_strategy
            )
        else:
            X_train_resampled = X_train_transformed
            y_train_resampled = self.y_train
        
        # Calculate class weight for imbalanced data
        pos_weight = len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1])
        
        # Build models dictionary
        self.models = {
            'Logistic_Regression': LogisticRegression(
                C=1.0, 
                class_weight='balanced',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='auc'
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                objective='binary',
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            
            'Neural_Network_MLP': Pipeline([
                ('scaler', StandardScaler()),  # Add scaler for better NN convergence
                ('mlp', LogisticRegression(  # Placeholder for testing, replace with actual NN in train_models
                    C=1.0, 
                    class_weight='balanced',
                    random_state=42
                ))
            ])
        }
        
        print("\nModel Configuration Complete:")
        for name in self.models.keys():
            print(f"- {name}")
        
        return self.models
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """
        Train a neural network model for churn prediction
        This would be replaced with a proper TensorFlow implementation in production
        """
        from sklearn.neural_network import MLPClassifier
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        # Train the model
        mlp.fit(X_train, y_train)
        
        return mlp
    
    def train_models(self, models=None, use_smote=True):
        """
        Train multiple models and store them
        """
        if models is None:
            models = self.models
        
        if models is None or len(models) == 0:
            self.build_models(use_smote=use_smote)
            models = self.models
            
        # Apply preprocessing
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        X_val_transformed = self.preprocessor.transform(self.X_val)
        
        # Apply SMOTE if requested
        if use_smote:
            X_train_resampled, y_train_resampled = self.apply_smote(
                sampling_strategy=0.8  # Target 80% of majority class
            )
        else:
            X_train_resampled = X_train_transformed
            y_train_resampled = self.y_train
        
        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Neural_Network_MLP':
                # For NN, replace the placeholder with actual NN
                models[name] = self.train_neural_network(
                    X_train_resampled, y_train_resampled,
                    X_val_transformed, self.y_val
                )
            else:
                model.fit(X_train_resampled, y_train_resampled)
            
            print(f"{name} training complete.")
        
        self.models = models
        return models
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, 
                      model_name=None, threshold=0.5, output_dir=None):
        """
        Evaluate model performance
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        if model_name is None:
            model_name = model.__class__.__name__
        
        # Preprocess the data (if raw features are provided)
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            X_train_eval = self.preprocessor.transform(X_train)
            X_val_eval = self.preprocessor.transform(X_val)
            X_test_eval = self.preprocessor.transform(X_test)
        else:
            X_train_eval = X_train
            X_val_eval = X_val
            X_test_eval = X_test
            
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train_eval)[:, 1]
            y_val_proba = model.predict_proba(X_val_eval)[:, 1]
            y_test_proba = model.predict_proba(X_test_eval)[:, 1]
            
            # Apply threshold for binary classification
            y_train_pred = (y_train_proba >= threshold).astype(int)
            y_val_pred = (y_val_proba >= threshold).astype(int)
            y_test_pred = (y_test_proba >= threshold).astype(int)
        else:
            y_train_pred = model.predict(X_train_eval)
            y_val_pred = model.predict(X_val_eval)
            y_test_pred = model.predict(X_test_eval)
            
            y_train_proba = y_train_pred
            y_val_proba = y_val_pred
            y_test_proba = y_test_pred
        
        # Calculate metrics
        results = {
            'Training': {
                'Accuracy': accuracy_score(y_train, y_train_pred),
                'Precision': precision_score(y_train, y_train_pred),
                'Recall': recall_score(y_train, y_train_pred),
                'F1 Score': f1_score(y_train, y_train_pred),
                'ROC AUC': roc_auc_score(y_train, y_train_proba),
                'Avg Precision': average_precision_score(y_train, y_train_proba)
            },
            'Validation': {
                'Accuracy': accuracy_score(y_val, y_val_pred),
                'Precision': precision_score(y_val, y_val_pred),
                'Recall': recall_score(y_val, y_val_pred),
                'F1 Score': f1_score(y_val, y_val_pred),
                'ROC AUC': roc_auc_score(y_val, y_val_proba),
                'Avg Precision': average_precision_score(y_val, y_val_proba)
            },
            'Test': {
                'Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred),
                'Recall': recall_score(y_test, y_test_pred),
                'F1 Score': f1_score(y_test, y_test_pred),
                'ROC AUC': roc_auc_score(y_test, y_test_proba),
                'Avg Precision': average_precision_score(y_test, y_test_proba)
            }
        }
        
        # Display performance metrics
        for dataset, metrics in results.items():
            print(f"\n{dataset} Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Classification report for test set
        print("\nClassification Report (Test Set):")
        class_report = classification_report(y_test, y_test_pred)
        print(class_report)
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["Test"]["ROC AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
        plt.close()
        
        # Generate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR Curve (AP = {results["Test"]["Avg Precision"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'))
        plt.close()
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # Save classification report as image
        plt.figure(figsize=(12, 8))
        plt.text(0.01, 0.99, class_report, fontsize=14, fontfamily='monospace', va='top')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_classification_report.png'))
        plt.close()
        
        # Only for tree-based models - feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])  # For linear models like logistic regression
        elif hasattr(model, 'feature_importances_') and hasattr(model, 'named_steps'):
            # For pipeline with feature importances
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    feature_importance = step.feature_importances_
                    break
        
        if feature_importance is not None:
            # Get feature names
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                except:
                    # Fallback if get_feature_names_out is not available
                    if X_train.shape[1] == len(feature_importance):
                        feature_names = X_train.columns
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            else:
                feature_names = X_train.columns if hasattr(X_train, 'columns') else [
                    f'Feature_{i}' for i in range(len(feature_importance))
                ]
            
            # Only use available feature names
            if len(feature_names) > len(feature_importance):
                feature_names = feature_names[:len(feature_importance)]
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Plot top N most important features
            top_n = min(20, len(importance_df))
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
            plt.title(f'Top {top_n} Feature Importances for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
            plt.close()
        
        # Generate threshold analysis plot
        thresholds = np.linspace(0.1, 0.9, 33)
        threshold_results = []
        
        for thresh in thresholds:
            thresh_pred = (y_val_proba >= thresh).astype(int)
            threshold_results.append({
                'Threshold': thresh,
                'Precision': precision_score(y_val, thresh_pred),
                'Recall': recall_score(y_val, thresh_pred),
                'F1': f1_score(y_val, thresh_pred)
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        plt.figure(figsize=(12, 8))
        plt.plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision')
        plt.plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall')
        plt.plot(threshold_df['Threshold'], threshold_df['F1'], label='F1 Score')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Current Threshold ({threshold:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis for {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_threshold_analysis.png'))
        plt.close()
        
        # For models like Random Forest, generate learning curve if applicable
        if hasattr(model, 'estimators_') and len(getattr(model, 'estimators_', [])) > 0:
            try:
                # Get validation scores at different training iterations
                if hasattr(model, 'evals_result'):
                    # For XGBoost
                    learning_curve_data = model.evals_result()
                elif hasattr(model, 'evals_result_'):
                    # For LightGBM
                    learning_curve_data = model.evals_result_
                else:
                    # For Random Forest - simulate by using subsets of trees
                    n_estimators = len(model.estimators_)
                    step_size = max(1, n_estimators // 10)
                    train_scores = []
                    val_scores = []
                    estimator_counts = list(range(step_size, n_estimators + step_size, step_size))
                    
                    for n in estimator_counts:
                        n_actual = min(n, n_estimators)
                        # For Random Forest, predict with subset of trees
                        y_train_subset = np.mean([
                            estimator.predict(X_train_eval) for estimator 
                            in model.estimators_[:n_actual]
                        ], axis=0)
                        y_val_subset = np.mean([
                            estimator.predict(X_val_eval) for estimator 
                            in model.estimators_[:n_actual]
                        ], axis=0)
                        
                        train_scores.append(f1_score(y_train, y_train_subset > 0.5))
                        val_scores.append(f1_score(y_val, y_val_subset > 0.5))
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(estimator_counts, train_scores, label='Training F1 Score')
                    plt.plot(estimator_counts, val_scores, label='Validation F1 Score')
                    plt.xlabel('Number of Estimators')
                    plt.ylabel('F1 Score')
                    plt.title(f'Learning Curve for {model_name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'{model_name}_learning_curve.png'))
                    plt.close()
            except Exception as e:
                print(f"Could not generate learning curve for {model_name}: {e}")
        
        return results, y_val_proba, y_test_proba
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Find the optimal threshold that maximizes F1 score
        """
        thresholds = np.linspace(0.1, 0.9, 33)  # More granular threshold search
        best_f1 = 0
        best_threshold = 0.5
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            
            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        
        print(f"\nOptimal threshold: {best_threshold:.3f}")
        print(f"At this threshold: Precision = {results_df.loc[results_df['threshold'] == best_threshold, 'precision'].values[0]:.4f}, " + 
              f"Recall = {results_df.loc[results_df['threshold'] == best_threshold, 'recall'].values[0]:.4f}, " + 
              f"F1 Score = {best_f1:.4f}")
        
        # Plot threshold analysis
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
        plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
        plt.plot(results_df['threshold'], results_df['f1'], label='F1 Score')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Optimization Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'optimal_threshold_analysis.png'))
        plt.close()
        
        return best_threshold, results_df
    
    def evaluate_all_models(self):
        """
        Evaluate all models and find the best one
        """
        if not self.models:
            raise ValueError("No models have been trained. Please train models first.")
        
        results = {}
        val_probas = {}
        test_probas = {}
        
        print("\n--- EVALUATING ALL MODELS ---")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            model_results, val_proba, test_proba = self.evaluate_model(
                model, self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test, 
                model_name=name
            )
            
            results[name] = model_results
            val_probas[name] = val_proba
            test_probas[name] = test_proba
            
            print(f"\nFinding optimal threshold for {name}...")
            optimal_threshold, _ = self.find_optimal_threshold(self.y_val, val_proba)
            
            # Re-evaluate with optimal threshold
            print(f"\n{name} with optimal threshold:")
            optimal_results, _, _ = self.evaluate_model(
                model, self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test, 
                model_name=f"{name}_Optimized",
                threshold=optimal_threshold
            )
            
            results[f"{name}_Optimized"] = optimal_results
        
        # Store the results
        self.model_results = results
        
        # Create model comparison table
        model_comparison = []
        for name, result in results.items():
            model_comparison.append({
                'Model': name,
                'Validation F1': result['Validation']['F1 Score'],
                'Test F1': result['Test']['F1 Score'],
                'Test Precision': result['Test']['Precision'],
                'Test Recall': result['Test']['Recall'],
                'Test AUC': result['Test']['ROC AUC']
            })
        
        comparison_df = pd.DataFrame(model_comparison).sort_values('Test F1', ascending=False)
        
        print("\n--- MODEL COMPARISON ---")
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        # Identify the best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        best_base_name = self.best_model_name.split('_Optimized')[0]
        self.best_model = self.models[best_base_name]
        self.best_threshold = self.find_optimal_threshold(self.y_val, val_probas[best_base_name])[0]
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"Best threshold: {self.best_threshold:.4f}")
        
        return comparison_df
    
    def create_ensemble(self, models=None, weights=None):
        """
        Create a weighted ensemble model from the trained models
        """
        if models is None:
            models = self.models
            
        if len(models) < 2:
            raise ValueError("Need at least 2 models to create an ensemble")
            
        # If weights are not specified, use validation F1 scores
        if weights is None:
            # Evaluate each model to get validation scores
            weights = []
            for name, model in models.items():
                if not hasattr(model, 'predict_proba'):
                    continue
                
                # Preprocess validation data
                X_val_transformed = self.preprocessor.transform(self.X_val)
                
                # Get validation probabilities
                val_proba = model.predict_proba(X_val_transformed)[:, 1]
                
                # Find optimal threshold
                threshold, _ = self.find_optimal_threshold(self.y_val, val_proba)
                
                # Apply threshold
                val_pred = (val_proba >= threshold).astype(int)
                
                # Calculate F1 score
                f1 = f1_score(self.y_val, val_pred)
                weights.append(f1)
        
        # Create the voting classifier with soft voting
        voting_models = [(name, model) for name, model in models.items() 
                         if hasattr(model, 'predict_proba')]
        
        if not voting_models:
            raise ValueError("None of the provided models support probability predictions")
            
        voting_classifier = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            weights=weights
        )
        
        # Train the voting classifier
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        voting_classifier.fit(X_train_transformed, self.y_train)
        
        return voting_classifier
    
    def train_stacking_ensemble(self):
        """
        Train a stacking ensemble model
        """
        # Get base estimators that support probability predictions
        base_estimators = [(name, model) for name, model in self.models.items() 
                          if hasattr(model, 'predict_proba')]
        
        if len(base_estimators) < 2:
            raise ValueError("Need at least 2 models for stacking")
            
        # Create stacking classifier with logistic regression meta-model
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1.0, class_weight='balanced'),
            cv=5,
            n_jobs=-1
        )
        
        # Train the stacking classifier
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        stacking_clf.fit(X_train_transformed, self.y_train)
        
        # Add to models dictionary
        self.models['Stacking_Ensemble'] = stacking_clf
        
        return stacking_clf
    
    def save_model(self, model=None, filename=None):
        """
        Save the model and preprocessor to disk
        """
        if model is None:
            model = self.best_model
            
        if filename is None:
            filename = 'best_churn_model.joblib'
            
        # Create full path
        path = os.path.join(self.output_dir, filename)
        
        # Save model with preprocessor and threshold
        dump({
            'model': model,
            'preprocessor': self.preprocessor,
            'threshold': self.best_threshold,
            'feature_columns': self.X_train.columns.tolist(),
            'categorical_columns': self.categorical_cols,
            'numerical_columns': self.numerical_cols
        }, path)
        
        print(f"Model saved to {path}")
        return path
    
    def load_model(self, filename):
        """
        Load a saved model
        """
        try:
            path = os.path.join(self.output_dir, filename)
            loaded = load(path)
            
            self.best_model = loaded['model']
            self.preprocessor = loaded['preprocessor']
            self.best_threshold = loaded['threshold']
            
            print(f"Model loaded from {path}")
            return loaded
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, X, threshold=None):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No best model available. Please train models first.")
            
        if threshold is None:
            threshold = self.best_threshold
            
        # Ensure X has the expected columns
        expected_columns = self.X_train.columns.tolist()
        if isinstance(X, pd.DataFrame):
            # Check if we need to add engineered features
            if len(X.columns) < len(expected_columns):
                X = self.add_engineered_features(X)
                
            # Ensure column order matches
            X = X[expected_columns]
        
        # Preprocess the data
        X_transformed = self.preprocessor.transform(X)
        
        # Get probability predictions
        if hasattr(self.best_model, 'predict_proba'):
            y_proba = self.best_model.predict_proba(X_transformed)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = self.best_model.predict(X_transformed)
            y_proba = y_pred  # For non-probabilistic models
            
        return y_pred, y_proba
    
    def generate_business_recommendations(self):
        """
        Generate business recommendations based on model insights
        """
        if hasattr(self, 'best_model') and self.best_model is not None:
            # Get feature importances if available
            feature_importance = None
            
            # Try to get feature importance from various model types
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                feature_importance = np.abs(self.best_model.coef_[0])
                
            if feature_importance is not None:
                # Get feature names (depending on model type and preprocessor)
                feature_names = None
                if hasattr(self.X_train, 'columns'):
                    feature_names = self.X_train.columns
                elif hasattr(self.preprocessor, 'get_feature_names_out'):
                    try:
                        feature_names = self.preprocessor.get_feature_names_out()
                    except:
                        pass
                
                if feature_names is not None and len(feature_names) == len(feature_importance):
                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Get top features
                    top_features = importance_df.head(10)['Feature'].tolist()
                    
                    # Generate recommendations based on top features
                    print("\n--- BUSINESS RECOMMENDATIONS ---")
                    print("\n1. Focus on these key factors affecting customer churn:")
                    
                    for i, feature in enumerate(top_features[:5]):
                        print(f"   {i+1}. {feature}")
                    
                    print("\n2. Customer Segmentation Strategies:")
                    print("   - High Risk (>70% churn probability): Immediate intervention with personalized retention offers")
                    print("   - Medium Risk (30-70% churn probability): Proactive engagement and satisfaction surveys")
                    print("   - Low Risk (<30% churn probability): Regular engagement and cross-selling opportunities")
                    
                    print("\n3. Targeted Interventions:")
                    
                    # Generate specific recommendations based on important features
                    if any('balance' in feature.lower() for feature in top_features):
                        print("   - For customers with high balance but low activity: Re-engagement campaigns")
                        
                    if any('product' in feature.lower() for feature in top_features):
                        print("   - For customers with multiple products but low activity: Product usage tutorials")
                        
                    if any('inactive' in feature.lower() for feature in top_features) or any('active' in feature.lower() for feature in top_features):
                        print("   - For inactive customers: Special activation incentives")
                        
                    if any('age' in feature.lower() for feature in top_features):
                        print("   - Age-based targeting: Customize communications and offers based on customer age groups")
                        
                    if any('credit' in feature.lower() for feature in top_features):
                        print("   - Credit score-based strategies: Offer credit limit increases to qualified customers")
                    
                    print("\n4. Monitoring and Continuous Improvement:")
                    print("   - Re-train the model quarterly with updated data")
                    print("   - A/B test different retention strategies")
                    print("   - Track key metrics: churn rate, retention costs, lifetime value")
                    
                    # Save recommendations to file
                    with open(os.path.join(self.output_dir, 'business_recommendations.txt'), 'w') as f:
                        f.write("BUSINESS RECOMMENDATIONS\n")
                        f.write("=======================\n\n")
                        f.write("1. Focus on these key factors affecting customer churn:\n")
                        for i, feature in enumerate(top_features[:5]):
                            f.write(f"   {i+1}. {feature}\n")
                        # Write rest of recommendations...
            
            return top_features if 'top_features' in locals() else None
        
        return None
    
    def run_full_pipeline(self, file_path=None, output_dir=None, use_smote=True):
        """
        Run the full modeling pipeline from data loading to business recommendations
        """
        if file_path:
            self.file_path = file_path
            
        if output_dir:
            self.output_dir = output_dir
            
        print("\n===== STARTING CHURN PREDICTION PIPELINE =====\n")
        
        # Step 1: Load data
        print("\n----- STEP 1: LOADING DATA -----")
        self.load_data()
        
        # Step 2: Exploratory data analysis
        print("\n----- STEP 2: EXPLORATORY DATA ANALYSIS -----")
        self.explore_data()
        
        # Step 3: Feature engineering
        print("\n----- STEP 3: FEATURE ENGINEERING -----")
        self.add_engineered_features()
        
        # Step 4: Data preparation
        print("\n----- STEP 4: DATA PREPARATION -----")
        self.prepare_data()
        
        # Step 5: Build models
        print("\n----- STEP 5: BUILDING MODELS -----")
        self.build_models(use_smote=use_smote)
        
        # Step 6: Train models
        print("\n----- STEP 6: TRAINING MODELS -----")
        self.train_models(use_smote=use_smote)
        
        # Step 7: Evaluate models
        print("\n----- STEP 7: EVALUATING MODELS -----")
        self.evaluate_all_models()
        
        # Step 8: Create ensemble (stacking)
        print("\n----- STEP 8: CREATING ENSEMBLE MODEL -----")
        try:
            stack = self.train_stacking_ensemble()
            print("\nEvaluating Stacking Ensemble...")
            self.evaluate_model(
                stack, self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test, 
                model_name="Stacking_Ensemble"
            )
        except Exception as e:
            print(f"Skipping stacking ensemble due to error: {e}")
        
        # Step 9: Save best model
        print("\n----- STEP 9: SAVING BEST MODEL -----")
        self.save_model(filename="best_churn_model.joblib")
        
        # Step 10: Generate business recommendations
        print("\n----- STEP 10: GENERATING BUSINESS RECOMMENDATIONS -----")
        self.generate_business_recommendations()
        
        print("\n===== CHURN PREDICTION PIPELINE COMPLETE =====")
        
        return self.best_model_name, self.best_threshold

# Example usage
if __name__ == "__main__":
    # Create the churn predictor
    predictor = BankChurnPredictor()
    
    # Run the full pipeline
    predictor.run_full_pipeline()
    
    # Alternatively, run steps individually:
    # predictor.load_data()
    # predictor.explore_data()
    # predictor.add_engineered_features()
    # predictor.prepare_data()
    # predictor.build_models()
    # predictor.train_models()
    # predictor.evaluate_all_models()
    # predictor.save_model()
    # predictor.generate_business_recommendations()