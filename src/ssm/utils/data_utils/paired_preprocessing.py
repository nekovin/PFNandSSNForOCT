from ssm.utils.data_utils.standard_preprocessing import standard_preprocessing
from ssm.utils.data_utils.oct_preprocessing import octa_preprocessing, remove_speckle_noise
from ssm.utils.data_utils.data_loading  import load_patient_data
import os
import re

def pair_data(preprocessed_data, octa_data, n_images_per_patient):
    n_neighbours = (len(preprocessed_data) - len(octa_data)) // 2
    
    input_target = []
    for i in range(len(octa_data)):
        oct_image = preprocessed_data[i + n_neighbours]
        octa_image = octa_data[i]
        input_target.append([oct_image, octa_image])
        
        if len(input_target) >= n_images_per_patient:
            break
            
    return input_target

def paired_octa_preprocessing(start=1, n_patients=1, n_images_per_patient=10, n_neighbours=2, threshold=0.65, sample = False, post_process_size=10):
    
    dataset = {}

    diabetes = 0
    base_data_path = os.environ["DATASET_DIR_PATH"]

    try:
        for i in range(start, start+n_patients):
            if diabetes != 0:
                base_data_path =  + rf"{diabetes}\RawDataQA-{diabetes} ({i})"
            else:
                data_path = base_data_path + rf"0\RawDataQA ({i})"
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
    
def _paired_preprocessing(start=1, n_patients=1, diabetes_list=[0,1,2], sample=False):
    """
    Preprocess OCT images without OCTA calculation, maintaining paired structure.
    """
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]
    
    try:
        for diabetes in diabetes_list:
            for i in range(start, start+n_patients):
                if diabetes != 0:
                    data_path = base_data_path + rf"{diabetes}\RawDataQA-{diabetes} ({i})"
                else:
                    data_path = base_data_path + rf"0\RawDataQA ({i})"
                print(f"Processing patient {i}")
                data = load_patient_data(data_path)
                print(f"Loaded {len(data)} images for patient {i}")
                assert len(data) > 0, f"No data found for patient {i}"
                
                preprocessed_data = standard_preprocessing(data)
                
                input_target = []
                for j in range(len(preprocessed_data)-1):
                    image1 = preprocessed_data[j]
                    image2 = preprocessed_data[j+1]
                    input_target.append([image1, image2])
                
                dataset[i] = input_target
            
        return dataset
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None



def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return 0 

def paired_preprocessing(start=1, n_patients=1, diabetes_list=[0,1,2], sample=False):
    """
    Preprocess OCT images without OCTA calculation, maintaining paired structure.
    """
    dataset = {}
    base_data_path = os.environ["DATASET_DIR_PATH"]
    
    patient_count = 0  # Counter for total patients processed
    try:
        for diabetes in diabetes_list:
            diabetes_path = os.path.join(base_data_path, f"{diabetes}")
            patient_dirs = sorted(os.listdir(diabetes_path), key=extract_number)
            sorted_patients = [os.path.join(diabetes_path, patient) for patient in patient_dirs]
            
            for i, patient_path in enumerate(sorted_patients, 1):
                    
                if patient_count >= n_patients:
                    break
                    
                data = load_patient_data(patient_path)
                print(f"Loaded {len(data)} images for patient {i}")
                assert len(data) > 0, f"No data found for patient {i}"
                
                preprocessed_data = standard_preprocessing(data)

                if len(preprocessed_data) <= 1: 
                    print(f"Warning: Patient {i} has insufficient images ({len(preprocessed_data)})")
                    continue
                
                input_target = []
                for j in range(len(preprocessed_data)-1):
                    image1 = preprocessed_data[j]
                    image2 = preprocessed_data[j+1]
                    input_target.append([image1, image2])
                
                dataset[i] = input_target
                patient_count += 1  
            
            if patient_count >= n_patients:
                break
                
        return dataset
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None