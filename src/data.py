"""Data loading and processing functions."""

import pandas as pd
from typing import Union


def load_unified_dataset(path_or_df):
    import pandas as pd
    import numpy as np

    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)

    # Datas e ordenação
    df['week_start_date'] = pd.to_datetime(df['week_start_date'], errors='coerce')
    df = df.dropna(subset=['week_start_date']).sort_values(['cd_mun', 'week_start_date']).reset_index(drop=True)

    # Tipos numéricos essenciais
    num_cols = [
        'morbidity_rate','mortality_rate','poverty_index','urbanization_index',
        'min_humidity','monthly_precipitation','demographic_density','max_temperature'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(np.float32)

    # Chave municipal
    df['cd_mun'] = pd.to_numeric(df['cd_mun'], errors='coerce').astype('Int64').astype('int64')

    return df


def load_neighbor_map(path_or_df):
    import pandas as pd
    map_df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)

    map_df['cd_mun'] = pd.to_numeric(map_df['cd_mun'], errors='coerce').astype('Int64').astype('int64')
    map_df['neighbor_cd_mun'] = pd.to_numeric(map_df['neighbor_cd_mun'], errors='coerce').astype('Int64').astype('int64')
    map_df = map_df.dropna().drop_duplicates().sort_values(['cd_mun', 'neighbor_cd_mun']).reset_index(drop=True)
    return map_df



def get_last_full_calendar_year(df: pd.DataFrame, date_col: str = 'week_start_date') -> int:
    """Get the last full calendar year present in the dataset.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        Last full calendar year as integer
    """
    df_dates = pd.to_datetime(df[date_col])
    years = df_dates.dt.year
    year_counts = years.value_counts()
    
    # Find last year with reasonable number of weeks (at least 50)
    for year in sorted(year_counts.index, reverse=True):
        if year_counts[year] >= 50:  # Should have ~52 weeks per year
            return year
    
    raise ValueError("No full calendar year found in dataset")


def slice_city(df: pd.DataFrame, city_id: int) -> pd.DataFrame:
    """Return one-city weekly DataFrame.
    
    Args:
        df: Full dataset
        city_id: Municipality ID
        
    Returns:
        DataFrame for single city, sorted by week_start_date
    """
    city_df = df[df['cd_mun'] == city_id].copy()
    if city_df.empty:
        raise ValueError(f"City {city_id} not found in dataset")
    
    return city_df.sort_values('week_start_date').reset_index(drop=True)


def create_synthetic_dataset(n_cities: int = 5, n_weeks: int = 520) -> pd.DataFrame:
    """Create synthetic dataset for testing.
    
    Args:
        n_cities: Number of cities
        n_weeks: Number of weeks per city
        
    Returns:
        Synthetic dataset matching expected schema
    """
    import numpy as np
    
    np.random.seed(42)
    cities = [1100015, 1100023, 1100031, 1100049, 1100056][:n_cities]
    
    data = []
    start_date = pd.Timestamp('1999-01-04')
    
    for city_id in cities:
        for week in range(n_weeks):
            date = start_date + pd.Timedelta(weeks=week)
            
            # Generate synthetic data with some seasonality
            week_of_year = date.isocalendar()[1]
            seasonal_factor = np.sin(2 * np.pi * week_of_year / 52)
            
            morbidity_rate = max(0, 5 + 3 * seasonal_factor + np.random.normal(0, 2))
            mortality_rate = max(0, morbidity_rate * 0.02 + np.random.normal(0, 0.1))
            
            row = {
                'cd_mun': city_id,
                'week_start_date': date,
                'morbidity_rate': morbidity_rate,
                'mortality_rate': mortality_rate,
                'poverty_index': np.random.uniform(1, 10),
                'urbanization_index': np.random.uniform(1, 10),
                'min_humidity': np.random.uniform(40, 90),
                'monthly_precipitation': np.random.uniform(0, 500),
                'demographic_density': np.random.uniform(1, 100),
                'max_temperature': np.random.uniform(20, 40)
            }
            data.append(row)
    
    return pd.DataFrame(data)


def create_synthetic_neighbor_map(cities: list) -> pd.DataFrame:
    """Create synthetic neighbor mapping for testing.
    
    Args:
        cities: List of city IDs
        
    Returns:
        Synthetic neighbor mapping
    """
    import itertools
    
    data = []
    # Create simple circular neighbor relationships
    for i, city in enumerate(cities):
        # Each city has 2 neighbors (previous and next in circular list)
        neighbor1 = cities[(i - 1) % len(cities)]
        neighbor2 = cities[(i + 1) % len(cities)]
        
        data.append({'cd_mun': city, 'neighbor_cd_mun': neighbor1})
        data.append({'cd_mun': city, 'neighbor_cd_mun': neighbor2})
    
    return pd.DataFrame(data)