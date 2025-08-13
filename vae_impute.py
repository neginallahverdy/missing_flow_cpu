import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from parse_model_args import ModelArgs
from dataset_def import CustomWeatherDataset
from HLVAE import HLVAE
from HL_VAE import read_functions

import os
import ast
from torch.utils.data import DataLoader
if __name__ == "__main__":
    opt = ModelArgs().parse_options()
    locals().update(opt)
    results_path = save_path + results_path
    gp_model_folder = save_path + gp_model_folder
    model_params = save_path + '/' + model_params
    folder_exists = os.path.isdir(results_path)
    if not folder_exists:
        raise "The resualt folder does not exist."

    if epochs not in [0, 1, 2] and not early_stopping:
        pd.to_pickle(opt,
                 os.path.join(save_path, 'arguments.pkl'))
    else:
        opt = pd.read_pickle(os.path.join(save_path, 'arguments.pkl'))
        opt['early_stopping'] = early_stopping
        opt['epochs'] = epochs
        opt['save_interval'] = save_interval
        opt['results_path'] = results_path
        opt['save_path'] = save_path
        opt['gp_model_folder'] = gp_model_folder
        opt['generate_images'] = generate_images
        opt['memory_dbg'] = memory_dbg
        opt['true_mask_file'] = true_mask_file
        opt['true_prediction_mask_file'] = true_prediction_mask_file
        opt['true_test_mask_file'] = true_test_mask_file
        opt['true_validation_mask_file'] = true_validation_mask_file
        opt['true_generation_mask_file'] = true_generation_mask_file
        if early_stopping:
            opt['model_params'] = os.path.join(save_path, 'early_best-vae_model.pth')
        else:
            opt['model_params'] = os.path.join(save_path, 'final-vae_model.pth')
        if 'ordinal' in save_path and 'convvae' in save_path:
            opt['vae_data_type'] = 'ordinal'
        locals().update(opt)
    dataset = CustomWeatherDataset(
        csv_file_data=csv_file_test_data,
        csv_file_label=csv_file_test_label,
        mask_file=test_mask_file,
        true_mask_file=true_test_mask_file,
        types_file=csv_types_file,
        id_covariate=id_covariate,
    )
    num_workers = 4
    setup_dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=num_workers)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

#     sys.path.append('.')

# # تنظیم مسیرها
# data_dir = Path('./data')
    model_path = Path('./results/final-vae_model.pth')
# device = torch.device("cpu")

# ایمپورت کلاس‌ها دقیقاً مشابه کد اصلی پروژه


# # مسیر فایل‌های موردنیاز
# files = {
#     "X": data_dir / "meteo_test_X.csv",
#     "Y": data_dir / "meteo_test_Y.csv",
#     "mask": data_dir / "mask_test.csv",
#     "types": data_dir / "data_types.csv",
# }

# # ساخت دیتاست (کاملاً مشابه فایل HLVAE_main.py)
# dataset = CustomWeatherDataset(
#     csv_file_data=str(files["X"]),
#     csv_file_label=str(files["Y"]),
#     mask_file=str(files["mask"]),
#     types_file=str(files["types"]),
#     device=device,
# )

    # ابعاد داده‌ها
    x_dim = dataset.cov_dim_ext
    y_dim = dataset.n_variables
    latent_dim = 32          # دقیقاً همان مقدار کانفیگ
    hidden_layers = ast.literal_eval(hidden_layers)

    # ساخت مدل دقیقاً مانند پروژه اصلی (اصلاح نهایی)
    vae = HLVAE([dataset.cov_dim_ext, hidden_layers, latent_dim, hidden_layers, y_dim], dataset.types_info,
                       dataset.n_variables, vy_init=[vy_init_real, vy_init_pos], logvar_network=logvar_network, conv=conv_hivae).to(
        device).to(torch.float64)

    # بارگذاری state_dict
    state_dict = torch.load(model_path, map_location=device)
    vae.load_state_dict(state_dict, strict=True)
    vae.eval()


    # پیش‌بینی مقادیر گم‌شده
    with torch.no_grad():
        batch = next(iter(setup_dataloader))
        print(
            "batch keys:", batch.keys(),
            "mask shape:", batch["mask"].shape,
            "data shape:" if "data" in batch else None,
        )

        data = batch["digit"].double().to(device)
        mask_full = batch["param_mask"].to(device)
        param_mask = batch["param_mask"].to(device)

        (
            p_samples,
            _,
            _,
            _,
            _,
            p_params,
            _,
            _,
        ) = vae(data, mask_full, param_mask, dataset.types_info)

        p_params_full = read_functions.p_params_concatenation_by_key(
            [p_params], dataset.types_info, data.shape[0], device, "x"
        )
        recon, _ = read_functions.statistics(
            p_params_full,
            dataset.types_info,
            device,
            log_vy=[vae._log_vy_real, vae._log_vy_pos],
        )
        recon = recon.cpu().numpy()
     
    # Debug: Check what's in types_info
    print("types_info keys:", list(dataset.types_info.keys()))
    print("types_info content:", dataset.types_info)
    
    # Extract pos columns from types_dict
    pos_cols = []
    if 'types_dict' in dataset.types_info:
        types_dict = dataset.types_info['types_dict']
        pos_cols = [i for i, type_info in enumerate(types_dict) if type_info['type'] == 'pos']
    else:
        print("Warning: Could not find types_dict in types_info")
        pos_cols = []
    
    if pos_cols:
        print(f"Applied exp transformation to {len(pos_cols)} pos columns: {pos_cols}")
    else:
        print("No pos columns found or transformed")

    # ذخیره خروجی‌ها
    output_path = results_path + "/Y_imputed_test.csv"

    pd.DataFrame(recon, columns=dataset.Y_df.columns).to_csv(output_path, index=False)

    # محاسبه MAE روی مقادیر ماسک شده
    miss = dataset.mask.cpu().numpy() == 0
    true_values = dataset.Y.cpu().numpy()
    mae = np.abs(recon[miss] - true_values[miss]).mean()
    print(f"MAE (hidden cells): {mae:.4f}")
    print(f"✅ Results saved to {output_path}")
