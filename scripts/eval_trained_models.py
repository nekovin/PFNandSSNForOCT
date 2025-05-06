from utils.evaluate import get_sample_image
from utils.data_loading import get_loaders
import torch
from utils.metrics import display_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2, evaluate_n2_with_ssm

# import rando
import random
import matplotlib.pyplot as plt

def plot_images(images, metrics_df):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i, (key, image) in enumerate(images.items()):
        #ax[i].imshow(image, cmap='gray')
        #ax[i].set_title(key)
        #ax[i].axis('off')
        ax[i // 4, i % 4].imshow(image, cmap='gray')
        ax[i // 4, i % 4].set_title(key)
        ax[i // 4, i % 4].axis('off')
        
    plt.show()

def main():
    #random.seed(42)
    start = random.randint(30, 35)
    train_loader, val_loader = get_loaders(start, 1, 50, 8)

    print("Loading data...")
    print("Patient ID: ", start)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = get_sample_image(val_loader, device)

    metrics = {}
    denoised_images = {
        "original": image.cpu().numpy()[0][0],
    }

    metrics, denoised_images = evaluate_n2(metrics, denoised_images, image)

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(image, device)
    metrics["pfn"] = prog_metrics
    denoised_images["pfn"] = prog_image

    metrics, denoised_images = evaluate_n2_with_ssm(metrics, denoised_images, image)


    metrics_df = display_metrics(metrics)

    def display_grouped_metrics(metrics):
        import pandas as pd
        from IPython.display import display, HTML
        
        # Define base schemas to group by
        base_schemas = ['n2n', 'n2v', 'n2s', 'pfn']
        
        schema_tables = {}
        
        # Process each metric key
        for metric_key in metrics.keys():
            # Find which base schema this metric belongs to
            matched_schema = None
            for schema in base_schemas:
                if schema in metric_key:
                    matched_schema = schema
                    break
            
            # If we found a matching schema
            if matched_schema:
                # Create the schema group if it doesn't exist
                if matched_schema not in schema_tables:
                    schema_tables[matched_schema] = {}
                
                # Add this metric column to the appropriate schema group
                schema_tables[matched_schema][metric_key] = metrics[metric_key]
        
        # Display tables for each schema group
        all_dfs = {}
        for schema, data in schema_tables.items():
            if data:  # Only process if we have data
                df = pd.DataFrame(data)
                
                # Apply styling to highlight maximum values in each row
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['font-weight: bold' if v else '' for v in is_max]
                
                styled_df = df.style.apply(highlight_max, axis=1)
                
                print(f"\n--- {schema.upper()} Models ---")
                display(styled_df)
                all_dfs[schema] = df
        
        return all_dfs
    
    display_grouped_metrics(metrics)
    
    plot_images(denoised_images, metrics_df)

if __name__ == "__main__":
    main()