import os
from ssm.trainers.n2_trainer import train_n2

def main(schema):
    
    patient_count = 5

    override_dict = {
        "training" : {
            "ablation": f"patient_count/{patient_count}_patients",
            "n_images_per_patient": 40,
            "n_patients" : patient_count,
            "method" : schema
            }
        }
    
    N2_PATH = os.environ.get("N2_CONFIG_PATH")

    train_n2(config_path=N2_PATH, schema=schema, ssm=True, override_config=override_dict)

if __name__ == "__main__":
    main()