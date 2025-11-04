from datasets import Dataset, load_dataset
from pathlib import Path
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt

def save_dataset_to_flat_hdf5(dataset, output_path, cuts_json):
    """
    A flat HDF5 file that can be directly loaded again via datasets.load_dataset('hdf5', ...).
    """

    # Convert to dictionary of numpy arrays
    data = {col: np.array(dataset[col]) for col in dataset.column_names}

    # Write to flat HDF5 structure
    with h5py.File(output_path, "w") as f:
        for col, arr in data.items():
            f.create_dataset(col, data=arr)
        f.attrs["sample_cuts"] = cuts_json

    print(f"✅ Saved dataset to {output_path}")

if __name__=="__main__":

    save_sample_on_disk = False
    base = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies_1000deg2_unlensed")
    
    fnames = list(base.glob("*"))
    fnames = sorted(fnames, key= lambda f:int(f.name.split(".")[0].split("-")[-1]), reverse=True)
    #print(fnames)
    #sys.exit()
    
    def generator(files, key_h5pyPath_pair, cuts=None):
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                data = {key:f[value][:] for key,value in key_h5pyPath_pair.items()}
    
                # Apply cuts here (before yielding to save memory)
                good = cuts(data)
                if cuts is not None:
                    mask = cuts(data)
                    for key in data:
                        #convert to numpy array right here.
                        data[key] = data[key][mask]
                for i in range(len(next(iter(data.values())))):
                    yield {key: data[key][i].item() for key in data}
    
    def cuts(data, generate_only_string=False):
        """manually fed function"""
        if generate_only_string:
            return "lsst_i<25.3"
        #mask = (np.array(data["lsst_i"][:])<25.3) & (np.array(data["redshift_true"][:])>=1) & (np.array(data["redshift_true"][:])<1.1)
        mask = (np.array(data["lsst_i"][:])<25.3)
        return mask
        
    
    dataset = Dataset.from_generator(generator, 
                                     gen_kwargs={
                                         "files":fnames,
                                         "key_h5pyPath_pair": {"lsst_i":"data/lsst_i",
                                                                "redshift_true":"data/redshift_true"},
                                         "cuts":cuts
                                         }
                                     )
    if save_sample_on_disk:
        json_cuts = cuts([], True)
        save_dataset_to_flat_hdf5(dataset, f"{json_cuts}_dataset.h5", cuts_json=json_cuts)

    print(dataset)
    print(dataset.column_names)

    ##try loading again
    #saved_dataset = load_dataset("hdf5", data_files=f"{json_cuts}_dataset.h5")
    ##saved_dataset = load_dataset("hdf5", data_files=f"lsst_i<25.3_dataset.h5")
    #print(saved_dataset)
    #print(saved_dataset.column_names)
    #sys.exit()

    # Now plot 
    print(type(dataset['lsst_i'][:]), type(dataset['redshift_true'][:]), )
    print(np.array(dataset['lsst_i'][:]).size, np.array(dataset['redshift_true'][:]).size, )
    print()

    rng = np.random.default_rng(2025)
    ids = rng.choice(np.arange(np.array(dataset['lsst_i'][:]).size), size=20000, replace=False)

    plt.scatter(np.array(dataset['redshift_true'][:])[ids], np.array(dataset['lsst_i'][:])[ids], s=1, alpha=0.2, label="lsst_i<25.3")
    plt.xlabel("z_true", fontsize=12)
    plt.ylabel("lsst_i", fontsize=12)
    plt.legend()
    plt.savefig("redshift_slice.png", bbox_inches="tight", dpi=120)

    plt.cla()
    plt.hist(np.array(dataset['redshift_true'][:])[ids], bins=310, label="lsst_i<25.3")
    plt.xlabel("z_true", fontsize=12)
    plt.ylabel("counts", fontsize=12)
    plt.legend()
    plt.savefig("redshift_dist.png", bbox_inches="tight", dpi=120)
    plt.close()
