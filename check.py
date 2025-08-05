import pandas as pd

df = pd.read_csv("payment_fraud.csv")
df = df.rename(columns={'label': 'isFraud'})

# Show 3 random fraud transactions
fraud_samples = df[df['isFraud'] == 1].sample(3, random_state=42)
print(fraud_samples)
