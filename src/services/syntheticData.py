import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset with n patients
n_patients = 350

data = {
    "Patient_ID": [f"P{str(i+1).zfill(3)}" for i in range(n_patients)],
    "Age": np.random.randint(50, 85, size=n_patients),
    # Gene counts from RNA-seq (random realistic expression counts)
    "Gene_JAK2": np.random.poisson(100, size=n_patients),
    "Gene_CALR": np.random.poisson(80, size=n_patients),
    "Gene_MPL": np.random.poisson(60, size=n_patients),
    "Gene_ASXL1": np.random.poisson(40, size=n_patients),
    "Gene_TET2": np.random.poisson(70, size=n_patients),
    "Gene_SRSF2": np.random.poisson(50, size=n_patients),
    # Clinical variables
    "WBC (10^9/L)": np.round(np.random.normal(12, 3, size=n_patients), 1),
    "Hgb (g/dL)": np.round(np.random.normal(10.5, 1.5, size=n_patients), 1),
    "Platelets (10^9/L)": np.round(np.random.normal(150, 40, size=n_patients), 0),
    "LDH (U/L)": np.round(np.random.normal(500, 100, size=n_patients), 0),
    "Spleen_Length (cm)": np.round(np.random.normal(16, 4, size=n_patients), 1),
    "Symptom_Score": np.round(np.random.uniform(2, 10, size=n_patients), 1),
    "Transfusion_Dependent": np.random.choice([0, 1], size=n_patients),
    # Target variable
    "Days_to_Event": np.random.randint(180, 2000, size=n_patients),
}

df = pd.DataFrame(data)
csv_path = "data/external/synthetic_myelofibrosis_data_n"+str(n_patients)+".csv"
df.to_csv(csv_path, index=False)

csv_path
