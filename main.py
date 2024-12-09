import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

# Load and prepare the training dataset
training_data = pd.read_csv('train.csv')

# Function to compute the geographical distance between two coordinates
def compute_distance(latitude1, longitude1, latitude2, longitude2):
    """
    Calculate the geographical distance (Haversine formula).
    Args:
        latitude1, longitude1: Coordinates of the first point.
        latitude2, longitude2: Coordinates of the second point.
    Returns:
        Distance in kilometers.
    """
    latitude1, longitude1, latitude2, longitude2 = map(np.radians, [latitude1, longitude1, latitude2, longitude2])
    delta_latitude = latitude2 - latitude1
    delta_longitude = longitude2 - longitude1
    formula = (
        np.sin(delta_latitude / 2) ** 2 +
        np.cos(latitude1) * np.cos(latitude2) * np.sin(delta_longitude / 2) ** 2
    )
    result = 2 * np.arcsin(np.sqrt(formula))
    earth_radius = 6371  # Earth radius in kilometers
    return result * earth_radius

# Clone and sort the data for processing
processed_data = training_data.copy()
processed_data.sort_values(by='unix_time', inplace=True)
grouped_by_card = processed_data.groupby('cc_num')

# Calculate new features based on distances
processed_data['distance_km'] = compute_distance(
    processed_data['lat'], processed_data['long'], processed_data['merch_lat'], processed_data['merch_long']
)

# Flags for similar categories in consecutive transactions
processed_data['previous_same_category'] = grouped_by_card['category'].shift(-1) == processed_data['category']
processed_data['next_same_category'] = grouped_by_card['category'].shift(1) == processed_data['category']
processed_data['category_consistency_flag'] = (
    processed_data['previous_same_category'] | processed_data['next_same_category']
)

# Compute average transaction times per category and time differences
processed_data['category_average_time'] = processed_data.groupby(['cc_num', 'category'])['unix_time'].transform('mean')
processed_data['category_time_difference'] = processed_data['unix_time'] - processed_data['category_average_time']

# Transaction timing metrics
processed_data['average_card_time'] = processed_data.groupby('cc_num')['unix_time'].diff().mean()
processed_data['time_from_previous_txn'] = processed_data['unix_time'] - processed_data['unix_time'].shift(1)
processed_data['time_from_previous_txn'].fillna(0, inplace=True)
processed_data['time_difference_metric'] = (
    processed_data['average_card_time'] - processed_data['time_from_previous_txn']
)

# Log-transform the transaction amount
processed_data['log_transaction_amount'] = np.log1p(processed_data['amt'])

# Count transactions per card and category
processed_data['txn_count_card'] = processed_data.groupby('cc_num')['cc_num'].transform('count')
processed_data['txn_count_card_category'] = processed_data.groupby(['cc_num', 'category'])['category'].transform('count')

# Compute rolling averages for transaction amounts
processed_data['rolling_average_10'] = (
    processed_data.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=10).mean()).fillna(processed_data['amt'])
)
processed_data['rolling_average_3'] = (
    processed_data.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window=3).mean()).fillna(processed_data['amt'])
)

# Average transaction amount by category
processed_data['average_amt_category'] = processed_data.groupby('category')['amt'].transform('mean')

# Transaction differences
processed_data['amount_difference_previous'] = grouped_by_card['amt'].diff().fillna(0)
processed_data['amount_difference_next'] = grouped_by_card['amt'].diff(-1).fillna(0)

# Identify high-value transactions
threshold_high_value = processed_data['amt'].quantile(0.9)
processed_data['high_value_transaction_flag'] = (processed_data['amt'] > threshold_high_value).astype(int)
processed_data['high_value_transaction_ratio'] = (
    processed_data.groupby('cc_num')['high_value_transaction_flag'].transform('mean')
)

# Extract hour from transaction time
processed_data['transaction_hour'] = pd.to_timedelta(processed_data['trans_time']).dt.total_seconds() / 3600

# Find the maximum time for each card
max_time_per_card = processed_data.loc[
    processed_data.groupby('cc_num')['rolling_average_3'].idxmax(), ['cc_num', 'unix_time']
]
max_time_per_card = max_time_per_card.drop_duplicates(subset='cc_num').set_index('cc_num')
processed_data['max_card_time'] = processed_data['cc_num'].map(max_time_per_card['unix_time'])

# Compute additional time-based metrics
processed_data['time_difference_max'] = abs(processed_data['unix_time'] - processed_data['max_card_time'])
processed_data['normalized_amount_time'] = processed_data['amt'] / (processed_data['time_difference_max'] + 1)

# Target variable
target_variable = 'is_fraud'

# Define feature and target data
X_features = processed_data.drop(
    columns=[
        target_variable, 'id', 'zip', 'average_card_time', 'state', 'long', 'lat', 'merch_lat', 'merch_long',
        'first', 'last', 'street', 'city', 'dob', 'merchant', 'job', 'trans_num', 'gender', 'time_from_previous_txn',
        'cc_num', 'city_pop', 'txn_count_card_category', 'time_difference_metric', 'category_average_time',
        'rolling_average_10', 'amt', 'next_same_category', 'distance_km', 'trans_time', 'max_card_time'
    ]
)
y_labels = processed_data[target_variable]

# Encode non-numeric columns
categorical_columns = X_features.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for column in categorical_columns:
    X_features[column] = encoder.fit_transform(X_features[column].astype(str))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
random_forest.fit(X_train, y_train)

# Evaluate the classifier
y_predictions = random_forest.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predictions))
print("Classification Report:")
print(classification_report(y_test, y_predictions))

# Feature importance
feature_importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_features.columns, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
