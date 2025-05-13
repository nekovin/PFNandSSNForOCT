from utils.standard_preprocessing import standard_preprocessing
from utils.oct_preprocessing import octa_preprocessing, remove_speckle_noise
from utils.data import load_patient_data

def pair_data(preprocessed_data, octa_data, n_images_per_patient):
    # Calculate the offset due to n_neighbours
    n_neighbours = (len(preprocessed_data) - len(octa_data)) // 2
    
    input_target = []
    for i in range(len(octa_data)):
        # Match each octa image with its corresponding OCT image
        # The octa at index i corresponds to the preprocessed at index i+n_neighbours
        oct_image = preprocessed_data[i + n_neighbours]
        octa_image = octa_data[i]
        input_target.append([oct_image, octa_image])
        
        if len(input_target) >= n_images_per_patient:
            break
            
    return input_target

def preprocessing_v2(start=1, n_patients=1, n_images_per_patient=10, n_neighbours=2, threshold=0.65, sample = False, post_process_size=10):
    
    dataset = {}

    diabetes = 0
    
    try:
        for i in range(start, start+n_patients):
            if diabetes != 0:
                data_path = rf"C:\Datasets\ICIP training data\ICIP training data\{diabetes}\RawDataQA-{diabetes} ({i})"
            else:
                data_path = rf"C:\Datasets\ICIP training data\ICIP training data\0\RawDataQA ({i})"
            print(f"Processing patient {i}")
            data = load_patient_data(data_path)
            assert len(data) > 0, f"No data found for patient {i}"
            
            preprocessed_data = standard_preprocessing(data)

            octa_data = octa_preprocessing(preprocessed_data, n_neighbours, threshold)
            
            cleaned_octa_data = []
            for octa_img in octa_data:
                cleaned_img = remove_speckle_noise(octa_img, min_size=post_process_size)
                cleaned_octa_data.append(cleaned_img)
            
            input_target_data = pair_data(preprocessed_data, cleaned_octa_data, n_images_per_patient)
            
            dataset[i] = input_target_data
            
        return dataset
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
    
    