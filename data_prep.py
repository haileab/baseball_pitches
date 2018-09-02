import pandas as pd
import numpy as np

from sklearn import preprocessing

def main():
    """
    Imports pitch data and returns a clean DF. Relies on helper functions at the bottom.
    """
    df = pd.read_csv('pitches', parse_dates = ['date'])
    df = df.drop(df[df.pitch_type.isnull()].index, axis = 0)
    df = df.drop(df[df['pitch_type'] == 'UN'].index, axis = 0)
    df.loc[:, 'pitch_type'] = df.apply(convert_pitch, axis = 1)
    df = meta_info(df)
    df.drop(['uid', 'game_pk', 'year', 'start_tfs_zulu', 'start_tfs'], axis = 1, inplace = True)
    #converting heights to inches
    df.loc[:, 'b_height_in'] = df.apply(height_in_inches, axis = 1)
    df = df.drop('b_height', axis=1)
    # on_1b, on_2b, on_3b have null values, it makes sense to replace the nans with zeros as it means no one is on base
    for col in ['on_1b', 'on_2b', 'on_3b']:
        df[col] = df[col].fillna(0)
    #renaming target variable
    df.rename(index=str, columns={"pitch_type": "target"}, inplace = True)
    #converting balls and strikes into a categorical category to see impact of an actual baseball scenerio such as 2 strikes 3 balls.
    df['balls_strikes'] = df.strikes.astype(str) + '-' + df.balls.astype(str)
    df.drop(['strikes', 'balls'], axis = 1, inplace=True)

    df, pitch_pct_columns = pitch_percentage(df)
    df, batting_pct_columns = at_bat_pct(df)

    features = ['team_id_b', 'team_id_p', 'inning', 'top', 'pcount_at_bat', 'pcount_pitcher', 'balls_strikes',
       'fouls', 'outs', 'batter_id', 'stand',
       'b_height_in', 'pitcher_id', 'p_throws', 'away_team_runs',
       'home_team_runs', 'on_1b', 'on_2b', 'on_3b', 'target']
    features = features + pitch_pct_columns + batting_pct_columns
    df = df[features]
    #label encoding object types with only 2 unique values (stand & p_throws)
    le = preprocessing.LabelEncoder()
    df['stand'] = le.fit_transform(df['stand'])
    df['p_throws'] = le.fit_transform(df['p_throws'])
    #le.fit(pitches)
    df['target'] = le.fit_transform(df['target'])

    dummy_cols = ['team_id_b', 'team_id_p', 'batter_id', 'pitcher_id', 'balls_strikes']
    df = pd.get_dummies(df, columns=dummy_cols)
    return df



####################
# Helper Functions
###################

def meta_info(df):
    """
    Imports metadata and keeps data prior to pitch.
    """
    metadata = pd.read_csv('pitch_by_pitch_metadata.csv', encoding='latin-1')
    not_avail = metadata[metadata.available_prior_to_pitch == 'No']
    not_avail = not_avail.column_name.tolist()
    not_avail.remove('pitch_type')
    not_avail = df[not_avail]
    avail_data = metadata[metadata.available_prior_to_pitch == 'Yes']
    avail_data = ['pitch_type', 'event'] + avail_data.column_name.tolist()
    return df[avail_data]


def convert_pitch(row):
    """
    Helper function to convert pitches to different categories. Used in main function.
    """
    pitch_catg = {'FF': 'fastball','SL': 'slider','CU': 'curveball','SI': 'sinker','FC': 'cutter','FT': 'fastball','KC': 'curveball','CH': 'changeup','IN': 'purpose-pitch','KN': 'off-speed','FS': 'fastball','FA': 'fastball','PO': 'purpose-pitch','EP': 'off-speed','SC': 'off-speed','AB': 'purpose-pitch','FO': 'changeup'}
    return pitch_catg[row['pitch_type']]


def height_in_inches(row):
    """
    Convert height from string format to an integer in inches.
    """
    return(int(row['b_height'][0]) * 12 + abs(int(row['b_height'][-2:])))


def pitch_percentage(df):
    """
    Creates a pitcher pitch type percentage for each pitcher using past pitch data.
    """

    #one hot encoding to create dummy variables
    dum =  pd.get_dummies(df['target'])
    df = pd.concat([df, dum], axis=1)
    ohe_cols = dum.columns
    ohe_cols = dum.columns

    cuml_cols = []
    pct_cols = []
    for col in ohe_cols:
        cuml_col = col + 'cuml'
        cuml_cols.append(cuml_col)
        df[cuml_col] = df.groupby(['pitcher_id'])[col].cumsum()

    df['total_pitches'] = df[cuml_cols].sum(axis=1)

    #create percentage columns
    for col in  cuml_cols:
        pct_col = col[:-4] + '_pct'
        pct_cols.append(pct_col)
        df[pct_col] = df[col]/df['total_pitches']

    #shift to move data back one pitch to avoid leakage from current pitch
    for col in pct_cols:
        df[col] = df.groupby(['pitcher_id'])[col].apply(lambda x:  x.shift(1))

    #dropping all the columns that we don't need anymore
    df.drop(columns= cuml_cols, inplace = True)
    df.drop(columns= ohe_cols, inplace = True)

    #fill in nans
    df[pct_cols] = df[pct_cols].fillna(0)
    return df, pct_cols

def at_bat_pct(df):
    '''
    Creates an at bat result percentage for each batter.
    '''
    #hashmap for converting at bat event outcome.
    event_catg = {'Strikeout':'out',
                 'Pop Out':'out',
                 'Groundout':'out',
                 'Lineout':'out',
                 'Flyout':'out',
                 'Walk':'walk',
                 'Runner Out':'out',
                 'Home Run':'home_run',
                 'Double':'double',
                 'Single':'single',
                 'Sac Bunt':'neutral',
                 'Forceout':'out',
                 'Grounded Into DP':'out',
                 'Sac Fly':'neutral',
                 'Field Error':'neutral',
                 'Intent Walk':'walk',
                 'Fielders Choice':'out',
                 'Triple':'triple',
                 'Hit By Pitch':'walk',
                 'Bunt Groundout':'out',
                 'Fan interference':'neutral',
                 'Fielders Choice Out':'out',
                 'Batter Interference':'neutral',
                 'Double Play':'double',
                 'Triple Play':'triple',
                 'Bunt Pop Out':'out',
                 'Strikeout - DP': 'out',
                 'Sac Fly DP':'out',
                 'Catcher Interference':'neutral'}

    #conversion of at bat result from 28 categories to 7
    def convert_at_bat_result(row):
            return event_catg.get(row['event'], 'neutral')

    df.loc[:, 'bat_result'] = df.apply(convert_at_bat_result, axis = 1)

    #one hot encoding to create dummy variables for at bat result
    dum =  pd.get_dummies(df['bat_result'])
    dum[df['pcount_at_bat'] > 1] = 0
    df = pd.concat([df, dum], axis=1)

    ohe_cols = dum.columns
    cuml_cols = []
    pct_bat_cols = []
    for col in ohe_cols:
        cuml_col = col + 'cuml'
        cuml_cols.append(cuml_col)
        df[cuml_col] = df.groupby(['batter_id'])[col].cumsum() - 1
        df[cuml_col].replace(-1, 0, inplace=True)

    df[cuml_cols].replace(-1, 0, inplace=True)
    df['total_at_bat'] = df[cuml_cols].sum(axis=1)
    #editing for removal of one pitch so we are only consider total previous pitches
    df['total_at_bat'] = df['total_at_bat'] - 1
    df['total_at_bat'].replace(-1, 0, inplace=True)

    def safe_div(row):
        if row['total_at_bat'] == 0:
            return -1
        return row[col] / row['total_at_bat']

    #create percentage columns
    for col in  cuml_cols:
        pct_col = col[:-4] + '_pct'
        pct_bat_cols.append(pct_col)
        df[pct_col] = df.apply(safe_div, axis=1)
    #dropping all the columns that we don't need anymore
    df.drop(columns= cuml_cols, inplace = True)
    df.drop(columns= ohe_cols, inplace = True)
    return df, pct_bat_cols



if __name__ == "__main__":
    main()
