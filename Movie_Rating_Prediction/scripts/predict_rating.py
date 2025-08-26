import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Data Loading
# Load the data with the correct file path
df = pd.read_csv('IMDb Movies India.csv')

# Step 2: Data Cleaning and Preprocessing
# Drop rows where 'Rating' is null as it's the target variable
df_cleaned = df.dropna(subset=['Rating'])

# Clean and convert 'Year' column
df_cleaned['Year'] = df_cleaned['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Clean and convert 'Duration' column, replacing missing values with a placeholder
df_cleaned['Duration'] = df_cleaned['Duration'].str.replace(' min', '', regex=False)
df_cleaned['Duration'] = pd.to_numeric(df_cleaned['Duration'], errors='coerce')
df_cleaned['Duration'].fillna(df_cleaned['Duration'].median(), inplace=True)

# Clean and convert 'Votes' column
df_cleaned['Votes'] = df_cleaned['Votes'].str.replace(',', '', regex=False).astype(int)

# Handle missing values in categorical columns by replacing with 'Unknown'
df_cleaned['Genre'].fillna('Unknown', inplace=True)
df_cleaned['Director'].fillna('Unknown', inplace=True)
df_cleaned['Actor 1'].fillna('Unknown', inplace=True)
df_cleaned['Actor 2'].fillna('Unknown', inplace=True)
df_cleaned['Actor 3'].fillna('Unknown', inplace=True)

# Step 3: Feature Engineering and Encoding
# Handle the 'Genre' column by splitting and one-hot encoding
genres = df_cleaned['Genre'].str.split(', ')
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(genres)
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df_cleaned.index)
df_processed = pd.concat([df_cleaned, genre_df], axis=1)

# Drop original 'Genre' and 'Name'
df_processed.drop(columns=['Name', 'Genre'], inplace=True)

# Function to encode top N features
def get_top_n_encoded_features(df, column, n=150):
    top_n_values = df[column].value_counts().nlargest(n).index.tolist()
    df[f'is_top_{column}'] = df[column].apply(lambda x: x if x in top_n_values else 'Other')
    return pd.get_dummies(df, columns=[f'is_top_{column}'], prefix=column)

# Apply encoding to Director and Actor columns
df_processed = get_top_n_encoded_features(df_processed, 'Director')
df_processed = get_top_n_encoded_features(df_processed, 'Actor 1')
df_processed = get_top_n_encoded_features(df_processed, 'Actor 2')
df_processed = get_top_n_encoded_features(df_processed, 'Actor 3')

# Drop the original high-cardinality columns
df_processed.drop(columns=['Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)

# Step 4: Model Building and Evaluation
# Select features and target variable
X = df_processed.drop('Rating', axis=1)
y = df_processed['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) score: {r2}")