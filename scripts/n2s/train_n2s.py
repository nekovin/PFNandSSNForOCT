import os
from fpss.trainers.n2_trainer import train_n2
from fpss.utils.seed import set_seed
from fpss.utils.config import get_config

def main():
    
    N2_CONFIG_PATH = os.environ.get("N2_CONFIG_PATH")

    config = get_config(N2_CONFIG_PATH)
    
    patient_count = config['training']['n_patients']
    n_images = config['training']['n_images_per_patient']

    schema = "n2s"

    override_dict = {
        "training" : {
            "ablation": f"patient_count/{patient_count}_patients/{n_images}_images",
            "n_patients" : patient_count,
            "method" : schema
            }
        }
    
    N2_PATH = os.environ.get("N2_CONFIG_PATH")
    
    print(f"Training {schema} model")
    set_seed(42)
    train_n2(config_path=N2_PATH, schema=schema, ssm=False, override_config=override_dict)
    set_seed(42)
    train_n2(config_path=N2_PATH, schema=schema, ssm=True, override_config=override_dict)

if __name__ == "__main__":
    main()