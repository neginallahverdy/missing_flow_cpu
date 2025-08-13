"""
This script imputes missing values in the weather test dataset using a
pre-trained HL-VAE model. It uses the same argument parsing as HLVAE_main.py.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import ast

# Import custom modules from your project
from HLVAE import HLVAE
from dataset_def import CustomWeatherDataset
from parse_model_args import ModelArgs # Import the argument parser

if __name__ == "__main__":
    # --- 1. Load Configuration using the project's standard parser ---
    # This block is adapted from HLVAE_main.py to load all settings
    opt = ModelArgs().parse_options()
    # This line loads all parsed options as local variables (e.g., save_path, model_params)
    locals().update(opt)
    hidden_layers = ast.literal_eval(hidden_layers)

    # Reconstruct paths using the base 'save_path' for consistency
    results_path = os.path.join('./results')
    gp_model_folder = os.path.join(save_path, gp_model_folder)
    
    # Determine the correct model file to use based on the early_stopping flag
    if early_stopping:
        model_params = os.path.join(save_path, 'early_best-vae_model.pth')
    else:
        model_params = os.path.join(save_path, 'final-vae_model.pth')

    # Ensure the results directory exists
    os.makedirs(results_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Loading model from: {model_params}")

    # --- 2. Load Dataset ---
    # The variable names now match those created by locals().update(opt)
    test_dataset = CustomWeatherDataset(
            root_dir=data_source_path,
            csv_file_data=csv_file_validation_data,
            csv_file_label=csv_file_validation_label,
            mask_file=validation_mask_file,
            true_miss_file=true_validation_mask_file,
            types_file=csv_types_file,
        )
    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # --- 3. Load Trained VAE Model ---
    # The VAE parameters are now available as local variables
    nnet_model = HLVAE([dataset.cov_dim_ext, hidden_layers, latent_dim, hidden_layers, y_dim], dataset.types_info,
                       dataset.n_variables, vy_init=[vy_init_real, vy_init_pos], logvar_network=logvar_network, conv=conv_hivae).to(
        device).to(torch.float64)
    try:
        nnet_model.load_state_dict(torch.load(model_params, map_location=lambda storage, loc: storage))
        print('Loaded pre-trained values.')
    except:
        print('Did not load pre-trained values.')
    
    # --- 4. Perform Imputation ---
    with torch.no_grad():
        batch = next(iter(dataloader))
        batch_Y = batch["Y"].to(device).double()
        mask_Y = batch["mask"].to(device)
        batch_X = batch["X"].to(device).double()

        p_samples, _, _, _, _, _, _, _ = vae(data=batch_Y, mask=mask_Y, param_mask=mask_Y, types_info=test_dataset.types_info, X_list=[batch_X])
        
        reconstructed_data = p_samples['x']
        if isinstance(reconstructed_data, list):
            reconstructed_data = torch.cat(reconstructed_data, dim=1)

        imputed_np = reconstructed_data.cpu().numpy()

    # --- 5. Post-Process and Save Full Imputed File ---
    types_dict = test_dataset.types_info.get('types_dict', [])
    pos_cols = [i for i, type_info in enumerate(types_dict) if type_info.get('type') == 'pos']
    if pos_cols:
        EPS = 1e-6
        imputed_np[:, pos_cols] = np.exp(imputed_np[:, pos_cols]) - EPS
        print(f"Applied inverse transformation to {len(pos_cols)} positive columns.")

    imputed_df = pd.DataFrame(imputed_np, columns=test_dataset.Y_df.columns)
    output_file = os.path.join(results_path, "imputed_weather_test_data.csv")
    imputed_df.to_csv(output_file, index=False)
    
    print("-" * 50)
    print(f"✅ Full imputed data saved to:\n{output_file}")
    
    # --- 6. Create and Save Comparison File for Missing Values ---
    print("\nCreating a comparison file for missing values...")

    miss_mask = test_dataset.mask.cpu().numpy() == 0
    ground_truth_np = test_dataset.Y.cpu().numpy()
    column_names = test_dataset.Y_df.columns

    comparison_data = []
    for row_idx in range(imputed_np.shape[0]):
        for col_idx in range(imputed_np.shape[1]):
            if miss_mask[row_idx, col_idx]:
                comparison_data.append({
                    'row_index': row_idx,
                    'column_name': column_names[col_idx],
                    'ground_truth': ground_truth_np[row_idx, col_idx],
                    'imputed_value': imputed_np[row_idx, col_idx]
                })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['absolute_error'] = (comparison_df['imputed_value'] - comparison_df['ground_truth']).abs()
    
    comparison_output_file = os.path.join(results_path, "comparison_imputed_vs_truth.csv")
    comparison_df.to_csv(comparison_output_file, index=False)
    
    print(f"✅ Comparison file with {len(comparison_df)} missing values saved to:\n{comparison_output_file}")
    print("-" * 50)