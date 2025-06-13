import pandas as pd

# Load data
nifty = pd.read_csv("Niftweekly.csv")
sensex = pd.read_csv("SensexWeekly.csv")

for df in [nifty, sensex]:
    df.rename(columns={'Change %': 'Change'}, inplace=True)
    df.drop(columns=['Open', 'High', 'Low'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df['Change'] = df['Change'].str.replace('%', '', regex=False).astype(float)
    df['Price'] = df['Price'].str.replace(',', '', regex=False).astype(float)
    df['Year'] = df['Date'].dt.year
print(nifty.head())
print(sensex.head())
print(nifty.shape)
print(sensex.shape)
# Calculate the number of weeks with a fall greater than 3% for Nifty and Sensex
# Filter weeks where absolute change is less than 3%
nifty_less3 = nifty[nifty['Change'] < -3]
sensex_less3 = sensex[sensex['Change'] < -3]
print(nifty_less3.shape)
print(sensex_less3.shape)

# Group and count
nifty_count = nifty_less3.groupby('Year').size().reset_index(name='Nifty <3% Weeks')
sensex_count = sensex_less3.groupby('Year').size().reset_index(name='Sensex <3% Weeks')