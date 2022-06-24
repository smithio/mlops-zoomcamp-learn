import pickle
import pandas as pd
import argparse

CATEGORICAL = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df
    
    
def run(year, month, to_save=False):
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = df[['ride_id']]
    df_result['pred'] = y_pred

    print(f'Average prediction for {year:04d}-{month:02d} = {y_pred.mean():0.2f}')

    if to_save:
        output_file = './predictions_202102.csv'

        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input parameters')
    parser.add_argument('--year', help='A year for the input datafile', required=True)
    parser.add_argument('--month', help='A month for the input datafile', required=True)
    args = vars(parser.parse_args())

    run(int(args['year']), int(args['month']))