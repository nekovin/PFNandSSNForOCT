

from .preprocessing import preprocessing
from .preprocessing_v2 import preprocessing_v2
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

def load_stage_1_data(
        regular=False, num_patients=4, img_per_patient=50, threshold=80, n_neighbours=5, post_process_size=40
        ):

    n_patients = num_patients
    n = img_per_patient
    n_images_per_patient = max(10, n)


    if regular:
        dataset = preprocessing(n_patients, n_images_per_patient, n_neighbours = 2,  threshold=threshold) #n neighbours must be 2
        name = "regular"
    else:
        dataset = preprocessing_v2(
            n_patients, n_images_per_patient, n_neighbours = n_neighbours, 
            threshold=threshold, post_process_size=post_process_size)
        name = "v2"

    print(f"Dataset size: {len(dataset)}")

    '''
    patient = n_patients - 1

    print(f"Dataset size: {len(dataset[patient])}")

    sample = 11

    fig, ax = plt.subplots( 1,2, figsize=(10, 5))
    fig.suptitle("OCTA and Raw Image")
    ax[0].imshow(dataset[patient][sample][0], cmap='gray', norm=NoNorm())
    ax[0].set_title("Raw Image")
    ax[0].axis('off')
    ax[1].imshow(dataset[patient][sample][1], cmap='gray')
    ax[1].set_title("OCTA Image")
    ax[1].axis('off')
    plt.show()
    '''

    raw_data = []
    octa_data = []
    for patient in dataset.keys():
        for pair in dataset[patient]:
            raw_data.append(pair[0])
            octa_data.append(pair[1])
            
    return raw_data, octa_data, dataset, name