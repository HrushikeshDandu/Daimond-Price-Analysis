# Diamond Price Analysis using Python - Complete Fixed Code
# Compatible with Python 3.8.9

# 1. Import Libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Starting Diamond Price Analysis...")

# 2. Load the Dataset
try:
    data = pd.read_csv("diamonds.csv")
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
except FileNotFoundError:
    print("Error: diamonds.csv not found. Please download the dataset first.")
    exit()

# 3. Data Exploration
print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)
print("\nDataset Info:")
print(data.info())
print("\nDataset Description:")
print(data.describe())

# 4. Clean the Data
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Remove unnamed column if it exists
if "Unnamed: 0" in data.columns:
    data = data.drop("Unnamed: 0", axis=1)
    print("Removed 'Unnamed: 0' column")

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove rows with zero dimensions (invalid diamonds)
initial_shape = data.shape[0]
data = data[(data['x'] > 0) & (data['y'] > 0) & (data['z'] > 0)]
print(f"Removed {initial_shape - data.shape[0]} rows with zero dimensions")

# 5. Feature Engineering
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Calculate diamond size (volume)
data["size"] = data["x"] * data["y"] * data["z"]
print("Added 'size' column (volume = length × width × depth)")
print(f"Size range: {data['size'].min():.2f} to {data['size'].max():.2f}")

# 6. Data Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# Visualization 1: Carat vs Price
print("Creating Carat vs Price visualization...")
try:
    figure1 = px.scatter(data_frame=data, x="carat", y="price", 
                        size="depth", color="cut", trendline="ols",
                        title="Diamond Price vs Carat",
                        labels={"carat": "Carat Weight", "price": "Price ($)"})
except ImportError:
    print("Statsmodels not found - creating scatter plot without trendline")
    figure1 = px.scatter(data_frame=data, x="carat", y="price", 
                        size="depth", color="cut",
                        title="Diamond Price vs Carat",
                        labels={"carat": "Carat Weight", "price": "Price ($)"})
figure1.show()

# Visualization 2: Size vs Price
print("Creating Size vs Price visualization...")
try:
    figure2 = px.scatter(data_frame=data, x="size", y="price", 
                        size="size", color="cut", trendline="ols",
                        title="Diamond Price vs Size (Volume)",
                        labels={"size": "Size (Volume)", "price": "Price ($)"})
except ImportError:
    print("Statsmodels not found - creating scatter plot without trendline")
    figure2 = px.scatter(data_frame=data, x="size", y="price", 
                        size="size", color="cut",
                        title="Diamond Price vs Size (Volume)",
                        labels={"size": "Size (Volume)", "price": "Price ($)"})
figure2.show()

# Visualization 3: Price Distribution by Cut and Color
print("Creating Price Distribution by Cut and Color...")
figure3 = px.box(data, x="cut", y="price", color="color",
                title="Price Distribution by Cut and Color")
figure3.show()

# Visualization 4: Price Distribution by Cut and Clarity
print("Creating Price Distribution by Cut and Clarity...")
figure4 = px.box(data, x="cut", y="price", color="clarity",
                title="Price Distribution by Cut and Clarity")
figure4.show()

# 7. Correlation Analysis
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

# Select numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print("Correlation with Price (sorted):")
price_corr = correlation["price"].sort_values(ascending=False)
print(price_corr)

# 8. Prepare Data for Machine Learning
print("\n" + "="*50)
print("PREPARING DATA FOR MACHINE LEARNING")
print("="*50)

# Debug: Check unique values in categorical columns
print("Unique cut values:", sorted(data['cut'].unique()))
print("Unique color values:", sorted(data['color'].unique()))
print("Unique clarity values:", sorted(data['clarity'].unique()))

# Convert categorical 'cut' to numerical
cut_mapping = {"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5}
data["cut_encoded"] = data["cut"].map(cut_mapping)

# Handle any unmapped cut values
if data["cut_encoded"].isna().any():
    print(f"Warning: Found unmapped cut values. Filling with default (3=Good)")
    data["cut_encoded"] = data["cut_encoded"].fillna(3)

# Also encode color and clarity for better model performance
color_mapping = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
clarity_mapping = {"FL": 1, "IF": 2, "VVS1": 3, "VVS2": 4, "VS1": 5, "VS2": 6, "SI1": 7, "SI2": 8, "I1": 9}

data["color_encoded"] = data["color"].map(color_mapping)
data["clarity_encoded"] = data["clarity"].map(clarity_mapping)

# Handle any unmapped values (NaN) by filling with default values
if data["color_encoded"].isna().any():
    print(f"Warning: Found unmapped color values. Filling with default (4=G)")
    data["color_encoded"] = data["color_encoded"].fillna(4)

if data["clarity_encoded"].isna().any():
    print(f"Warning: Found unmapped clarity values. Filling with default (6=VS2)")
    data["clarity_encoded"] = data["clarity_encoded"].fillna(6)

# Check for any remaining NaN values in our feature columns
print("\nChecking for NaN values in features:")
features_to_check = ["carat", "cut_encoded", "color_encoded", "clarity_encoded", "depth", "table", "size"]
for feature in features_to_check:
    nan_count = data[feature].isna().sum()
    if nan_count > 0:
        print(f"  {feature}: {nan_count} NaN values")
        # Fill NaN values with median for numeric columns
        if data[feature].dtype in ['float64', 'int64']:
            median_val = data[feature].median()
            data[feature] = data[feature].fillna(median_val)
            print(f"    Filled with median: {median_val}")
    else:
        print(f"  {feature}: No NaN values ✓")

# Select features for the model
features = ["carat", "cut_encoded", "color_encoded", "clarity_encoded", "depth", "table", "size"]
X = data[features].values
y = data["price"].values

# Final check for NaN values in training data
print(f"\nFeatures selected: {features}")
print(f"Training data shape: X={X.shape}, y={y.shape}")
print(f"NaN values in X: {np.isnan(X).sum()}")
print(f"NaN values in y: {np.isnan(y).sum()}")

# Remove any rows with NaN values if they still exist
if np.isnan(X).any() or np.isnan(y).any():
    print("Removing rows with NaN values...")
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    print(f"After NaN removal: X={X.shape}, y={y.shape}")

# 9. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 10. Train Machine Learning Model
print("\n" + "="*50)
print("TRAINING MACHINE LEARNING MODEL")
print("="*50)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training RandomForestRegressor...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 11. Diamond Price Prediction Function
print("\n" + "="*50)
print("DIAMOND PRICE PREDICTION")
print("="*50)

def predict_diamond_price():
    """Interactive function to predict diamond price"""
    print("\nDiamond Price Prediction")
    print("-" * 30)
    
    try:
        # Get user input
        carat = float(input("Enter Carat Size (e.g., 1.5): "))
        print("Cut Types: 1=Ideal, 2=Premium, 3=Good, 4=Very Good, 5=Fair")
        cut = int(input("Enter Cut Type (1-5): "))
        print("Color: 1=D (best), 2=E, 3=F, 4=G, 5=H, 6=I, 7=J (worst)")
        color = int(input("Enter Color (1-7): "))
        print("Clarity: 1=FL (best), 2=IF, 3=VVS1, 4=VVS2, 5=VS1, 6=VS2, 7=SI1, 8=SI2, 9=I1 (worst)")
        clarity = int(input("Enter Clarity (1-9): "))
        depth = float(input("Enter Depth percentage (e.g., 62.5): "))
        table = float(input("Enter Table percentage (e.g., 57.0): "))
        size = float(input("Enter Size/Volume (e.g., 25.0): "))
        
        # Create feature array
        features_input = np.array([[carat, cut, color, clarity, depth, table, size]])
        
        # Make prediction
        predicted_price = model.predict(features_input)[0]
        
        print(f"\n💎 Predicted Diamond Price: ${predicted_price:.2f}")
        
        # Show confidence level
        if r2 > 0.8:
            print("🟢 High confidence prediction")
        elif r2 > 0.6:
            print("🟡 Medium confidence prediction")
        else:
            print("🔴 Low confidence prediction")
            
    except ValueError:
        print("❌ Please enter valid numbers")
    except Exception as e:
        print(f"❌ Error: {e}")

# Example prediction without user input (for testing)
print("\nExample Prediction:")
print("Predicting price for a 1.0 carat, Ideal cut, D color, VVS1 clarity diamond...")
example_features = np.array([[1.0, 1, 1, 3, 62.0, 57.0, 25.0]])
example_price = model.predict(example_features)[0]
print(f"Example predicted price: ${example_price:.2f}")

# Run the interactive prediction function
predict_diamond_price()

# 12. Save Results
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Save processed data
data.to_csv("diamonds_processed.csv", index=False)
print("✅ Processed data saved to 'diamonds_processed.csv'")

# Save model predictions
results_df = pd.DataFrame({
    'actual_price': y_test,
    'predicted_price': y_pred,
    'difference': y_test - y_pred
})
results_df.to_csv("model_predictions.csv", index=False)
print("✅ Model predictions saved to 'model_predictions.csv'")

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)
print("✅ Feature importance saved to 'feature_importance.csv'")

print("\n🎉 Diamond Price Analysis Complete!")
print(f"📊 Model Accuracy: {r2:.1%}")
print("📁 Check the generated CSV files for detailed results")
print(f"📈 The model can predict diamond prices with {r2:.1%} accuracy")