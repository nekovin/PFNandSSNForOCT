def evaluate():
    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint
    checkpoint_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\notebooks\models\nonlocal_gan_epoch_100.pth"
    start_epoch = load_model(generator, discriminator, optimizer_g, optimizer_d, checkpoint_path)

    print(f"Loaded model from epoch {start_epoch}")

    start = 10
    n_patients = 1
    n_images_per_patient = 20
    batch_size = 8

    train_loader, val_loader = get_loaders(start, n_patients, n_images_per_patient, batch_size)

    evaluate_model(generator, val_loader, save_dir='results')

    #evaluate_model(generator, test_loader)