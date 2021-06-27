import click
import pandas as pd
from atom3d.datasets.datasets import LMDBDataset, make_lmdb_dataset

class UpdateTypes():
    def __init__(self, df_keys):
        self.df_keys = df_keys
    def __call__(self,x):
        if 'types' not in x.keys():
            for key in self.df_keys:
                if key in x and type(x[key]) != pd.DataFrame:
                    x[key] = pd.DataFrame(**x[key])
        return x

@click.command()
@click.argument('old_name', type=click.Path(exists=True))
@click.argument('new_name', type=click.Path(exists=False))
@click.argument('df_keys',  type=str, nargs=-1)
def main(old_name, new_name, df_keys):
    dataset = LMDBDataset(old_name, transform=UpdateTypes(df_keys))
    make_lmdb_dataset(dataset, new_name)
    dataset_new = LMDBDataset(new_name)

if __name__ == "__main__":
    main()

