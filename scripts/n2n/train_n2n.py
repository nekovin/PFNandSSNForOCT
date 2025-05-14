import os
from ssm.trainers.n2_trainer import train_n2
from ssm.utils.seed import set_seed

def main():
    
    patient_count = 25

    schema = "n2n"

    override_dict = {
        "training" : {
            "ablation": f"patient_count/{patient_count}_patients",
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