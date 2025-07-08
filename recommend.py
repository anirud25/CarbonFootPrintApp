import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Initialize S3 client
s3 = boto3.client('s3')

# Define the S3 bucket and file name
bucket_name = 'mydevawsbucket-23023'
file_name = 'lifestyle_sustainability_data.csv'

# Load the dataset
dataset = pd.read_csv(file_name)
print(dataset.shape)

#dataset = dataset.dropna()

# Recreate the label encoders for categorical variables
label_encoders = {}
for column in dataset.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])

# Mapping the encoded numeric values to their respective emission factors
emission_factors = {
    'EnergySource': {0: 0.02, 1: 0.25, 2: 0.5},  # Mapping the encoded values for 'Renewable', 'Mixed', 'Non-Renewable'
    'TransportationMode': {0: 0, 1: 0.05, 2: 0.24, 3: 0},  # Mapping for 'Bike', 'Public Transit', 'Car', 'Walk'
    #'DietType': {0: 1.5, 1: 2.5, 2: 5.0},  # Mapping for 'Mostly Plant-Based', 'Balanced', 'Mostly Animal-Based'
     'DietType': {
        0: 0.75,   # Mostly Plant-Based: 0.75 kg CO2 per meal (for 0.5 kg of food)
        1: 1.25,   # Balanced: 1.25 kg CO2 per meal (for 0.5 kg of food)
        2: 2.5     # Mostly Animal-Based: 2.5 kg CO2 per meal (for 0.5 kg of food)
    },
    'UsingPlasticProducts': {0: 1.0, 1: 2.5, 2: 5.0, 3: 0.5},  # Mapping for 'Rarely', 'Sometimes', 'Often', 'Never'
    'DisposalMethods': {0: 0.5, 1: 1.0, 2: 3.0, 3: 2.0},  # Mapping for 'Composting', 'Recycling', 'Landfill', 'Combination'
}

# Re-define the carbon footprint calculation function using the correct encoded values
def calculate_carbon_footprint(row):
    # Electricity
    energy_emission_factor = emission_factors['EnergySource'][row['EnergySource']]
    carbon_footprint_electricity = row['MonthlyElectricityConsumption'] * energy_emission_factor

    # Transportation
    transport_emission_factor = emission_factors['TransportationMode'][row['TransportationMode']]
    average_distance_per_month = 600  # Example assumption: 20 km per day * 30 days
    carbon_footprint_transport = average_distance_per_month * transport_emission_factor

    # Diet
    diet_emission_factor = emission_factors['DietType'][row['DietType']]
    average_meals_per_month = 90  # Example assumption: 3 meals/day * 30 days
    carbon_footprint_diet = average_meals_per_month * diet_emission_factor

    # Plastic Use
    plastic_emission_factor = emission_factors['UsingPlasticProducts'][row['UsingPlasticProducts']]
    carbon_footprint_plastic = plastic_emission_factor

    # Waste Disposal
    disposal_emission_factor = emission_factors['DisposalMethods'][row['DisposalMethods']]
    carbon_footprint_disposal = disposal_emission_factor

    # Total Carbon Footprint
    total_carbon_footprint = (
        carbon_footprint_electricity +
        carbon_footprint_transport +
        carbon_footprint_diet +
        carbon_footprint_plastic +
        carbon_footprint_disposal
    )

    return total_carbon_footprint

# Apply the updated carbon footprint calculation
dataset['CarbonFootprint'] = dataset.apply(calculate_carbon_footprint, axis=1)

# Display the dataset with the calculated carbon footprint
dataset[['ParticipantID', 'CarbonFootprint']].head()


def generate_recommendations(row):
    recommendations = []

    if row['EnergySource'] == 2:  # Non-Renewable
        recommendations.append("Consider switching to renewable energy sources to reduce emissions.")

    if row['TransportationMode'] == 2:  # Car
        recommendations.append("Opt for public transit, biking, or walking to lower transportation emissions.")

    if row['DietType'] == 2:  # Mostly Animal-Based
        recommendations.append("Shift towards a plant-based diet to significantly cut down on diet-related emissions.")

    if row['UsingPlasticProducts'] == 2:  # Often
        recommendations.append("Reduce plastic usage and choose sustainable alternatives.")

    if row['DisposalMethods'] == 2:  # Landfill
        recommendations.append("Adopt recycling and composting practices to minimize waste emissions.")

    return "; ".join(recommendations)

dataset['Recommendations'] = dataset.apply(generate_recommendations, axis=1)

# Display the dataset with recommendations
dataset[['ParticipantID', 'CarbonFootprint', 'Recommendations']].head()


print(dataset.head(), dataset.shape)

dataset.to_csv('lifestyle_sustainability_data.csv', index=False)