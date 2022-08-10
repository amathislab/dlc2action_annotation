import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import click


class FileTransformer:
    def __init__(self, labels_file):
        self.labels_file = labels_file
        name = self.labels_file.split(".")[0]
        self.output_file = name + ".pickle"

    def transform(self):
        success = self.load_labels()
        if success:
            self.save()

    def save(self):
        with open(self.output_file, "wb") as f:
            pickle.dump(
                (self.metadata, self.loaded_labels, self.animals, self.loaded_times), f
            )

    def load_labels(self, length=None):
        if self.labels_file is not None:
            try:
                with pd.HDFStore(self.labels_file) as store:
                    self.loaded_data = store["annotation"]
                    try:
                        self.metadata = store.get_storer("annotation").attrs.metadata
                    except:
                        self.metadata = {}
                self.loaded_labels = list(self.loaded_data.columns.unique("categories"))
                self.animals = list(self.loaded_data.columns.unique("individuals"))
                n_cat = len(self.loaded_data.columns.unique("categories"))
                self.loaded_data = self.loaded_data.to_numpy().T.reshape(
                    (self.n_ind, n_cat, -1)
                )
                times = [[] for i in range(self.n_ind)]
                from copy import copy

                for ind_i, ind_list in enumerate(self.loaded_data):
                    for i in range(ind_list.shape[0]):
                        l = copy(ind_list[i, :])
                        l = (l == 0.5).astype(int)
                        list_amb = [
                            [*x, True]
                            for x in np.flatnonzero(
                                np.diff(np.r_[0, l, 0]) != 0
                            ).reshape(-1, 2)
                        ]
                        l = copy(ind_list[i, :])
                        l = (l == 1).astype(int)
                        list_sure = [
                            [*x, False]
                            for x in np.flatnonzero(
                                np.diff(np.r_[0, l, 0]) != 0
                            ).reshape(-1, 2)
                        ]
                        times[ind_i].append(np.array(list_amb + list_sure))
                self.loaded_times = times
                del self.loaded_data
                return True
            except:
                try:
                    with open(self.labels_file, "rb") as f:
                        _ = pickle.load(f)
                    print(f"{self.labels_file} seems to be ok")
                except:
                    print(
                        f"something is wrong with {self.labels_file} (cannot be opened by pickle or hdf5)"
                    )
                return False


@click.command()
@click.option(
    "--directory", required=True, help="the directory where the files to transform are"
)
def main(directory):
    files = [file for file in os.listdir(directory) if file.split(".")[-1] == "h5"]
    for file in tqdm(files):
        fts = FileTransformer(os.path.join(directory, file))
        fts.transform()


if __name__ == "__main__":
    main()
