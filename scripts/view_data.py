import vigra
import numpy as np
from volumina_viewer import volumina_n_layer

def view():
    raw = vigra.readHDF5("../data/inp0.h5", 'data').astype('float32')
    seg = vigra.readHDF5("../data/seg0.h5", 'data')
    mc_res = vigra.readHDF5("../data/sample_B_test_mc_defectsV2.h5", 'data')

    volumina_n_layer([raw,seg,mc_res])

def extract_example():
    raw = vigra.readHDF5("../data/inp0.h5", 'data').astype('float32')
    seg = vigra.readHDF5("../data/seg0.h5", 'data')
    res = vigra.readHDF5("../data/sample_B_test_mc_defectsV2.h5", 'data')

    mask = np.s_[:768,:768,20]
    raw = raw[mask]
    seg = seg[mask]
    res = res[mask]

    vigra.writeHDF5(raw, "../data/example_data.h5", "raw")
    vigra.writeHDF5(seg, "../data/example_data.h5", "seg")
    vigra.writeHDF5(res, "../data/example_data.h5", "res")

    volumina_n_layer([raw, seg, res])

def view_example():
    raw = vigra.readHDF5("../data/example_data.h5", "raw")
    seg = vigra.readHDF5("../data/example_data.h5", "mergedseg")
    res = vigra.readHDF5("../data/example_data.h5", "res")

    volumina_n_layer([raw, seg, res])


if __name__ == '__main__':
    view_example()
