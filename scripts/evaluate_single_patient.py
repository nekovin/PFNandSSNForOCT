def main():
    import matplotlib.pyplot as plt
    from evaluation.evaluate_n2_baselines import evaluate_baseline, evaluate_ssm_constraint
    from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
    from utils.evaluate import load_sdoct_dataset
    from tqdm import tqdm
    import torch

    override_config = {
        "1" : ""
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)

    for patient_id, patient_data in tqdm(dataset.items(), desc="Evaluating patients"):
        raw_image = patient_data["raw"].to(device)
        reference = patient_data["avg"].to(device)[0][0]
        break


    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference.cpu().numpy(), cmap="gray")
    ax[1].set_title("Reference Image")
    plt.show()


    n2n_metrics, n2n_denoised = evaluate_baseline(raw_image, reference, "n2n")
    n2n_ssm_metrics, n2n_ssm_denoised = evaluate_ssm_constraint(raw_image, reference, "n2n")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2n_denoised, cmap="gray")
    ax[0].set_title("N2N Denoised")
    ax[1].imshow(n2n_ssm_denoised, cmap="gray")
    ax[1].set_title("N2N SSM Denoised")
    plt.show()

    n2v_metrics, n2v_denoised = evaluate_baseline(raw_image, reference, "n2v")
    n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(raw_image, reference, "n2v")


    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2v_denoised, cmap="gray")
    ax[0].set_title("N2V Denoised")
    ax[1].imshow(n2v_ssm_denoised, cmap="gray")
    ax[1].set_title("N2V SSM Denoised")
    plt.show()


    n2s_metrics, n2s_denoised = evaluate_baseline(raw_image, reference, "n2s")
    n2s_ssm_metrics, n2s_ssm_denoised = evaluate_ssm_constraint(raw_image, reference, "n2s")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2s_denoised, cmap="gray")
    ax[0].set_title("N2S Denoised")
    ax[1].imshow(n2s_ssm_denoised, cmap="gray")
    ax[1].set_title("N2S SSM Denoised")
    plt.show()

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(raw_image, reference, device)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(prog_image, cmap="gray")

    ax[0].set_title("Progressive Fusion Denoised")  
    plt.show()

    print("N2N Metrics: ", n2n_metrics)
    print("N2N SSM Metrics: ", n2n_ssm_metrics)
    print("N2V Metrics: ", n2v_metrics)
    print("N2V SSM Metrics: ", n2v_ssm_metrics)
    print("N2S Metrics: ", n2s_metrics)
    print("N2S SSM Metrics: ", n2s_ssm_metrics)
    print("Progressive Fusion Metrics: ", prog_metrics)

if __name__ == "__main__":
    main()