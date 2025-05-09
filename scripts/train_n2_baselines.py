from trainers.n2n_trainer import train_n2n
from trainers.n2v_trainer import train_n2v
from trainers.n2s_trainer import train_n2s

def main():
    schemas = {
        'n2n': train_n2n,
        'n2v': train_n2v,
        'n2s': train_n2s
    }

    patient_count = 40

    override_dict = {
        "training" : {
            "ablation": f"baselines/{patient_count}_patients",
            "n_images_per_patient": 20,
            "n_patients" : patient_count,
            "load" : True
            }
        }

    for schema in schemas.keys():
        print(f"Training {schema} model")
        schemas[schema](r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml", False, override_dict)

if __name__ == "__main__":
    main()