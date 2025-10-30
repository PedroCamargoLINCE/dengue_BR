import pandas as pd

df = pd.read_csv('data/unified_dataset.csv')
print('Total municipalities:', df['cd_mun'].nunique())
print('\nData availability per city:')

cities = {
    'Florianópolis': 4205407, 
    'Porto Alegre': 4314902, 
    'Curitiba': 4106902, 
    'Vitória': 3205309
}

for name, code in cities.items():
    city_df = df[df['cd_mun'] == code]
    print(f'{name} ({code}): {len(city_df)} weeks')

# Calculate minimum required weeks
train_window_years = 5
test_window_weeks = 52
lookback_weeks = 104  # Max from configs
horizon_weeks = 4

min_weeks = (train_window_years * 52) + test_window_weeks + lookback_weeks + horizon_weeks
print(f'\nMinimum weeks required: {min_weeks}')
print(f'  - Train: {train_window_years * 52} weeks')
print(f'  - Test: {test_window_weeks} weeks')  
print(f'  - Lookback: {lookback_weeks} weeks')
print(f'  - Horizon: {horizon_weeks} weeks')
