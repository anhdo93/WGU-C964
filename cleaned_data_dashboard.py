import pandas as pd

def data():
    # Import cleaned data and remapping values
    df = pd.read_csv('./data/cleaned_data.csv')
    df.drop(columns=['ID','Unnamed: 0'], inplace=True)

    # Gender
    map_array = {'0':'Female', '1':'Male'}
    df['Gender'] = df['Gender'].astype(str).replace(map_array)
    # Car/Realty/Work Phone/Phone/Email
    map_array = {'0':'No', '1':'Yes'}
    df['Car'] = df['Car'].astype(str).replace(map_array)
    df['Realty'] = df['Realty'].astype(str).replace(map_array)
    df['Work Phone'] = df['Work Phone'].astype(str).replace(map_array)
    df['Phone'] = df['Phone'].astype(str).replace(map_array)
    df['Email'] = df['Email'].astype(str).replace(map_array)

    return df
