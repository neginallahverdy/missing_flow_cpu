def HLVAETest_with_imputation_save(test_dataset, nnet_model, save_path, prnt=True, test=False, id_covariate=2, T=20, training_indexes=[]):
    """
    Modified version of HLVAETest that saves ground truth vs imputed data to CSV
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Length of test dataset:  {}'.format(len(test_dataset)))
    dataloader_test = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=0)
    torch.cuda.empty_cache()
    
    p_params_test_list = []
    log_p_x_test_list = []
    log_p_x_test_missing_list = []
    indexes = np.array(range(0, len(test_dataset)))
    test_x = torch.DoubleTensor(test_dataset.label_source.values).to(device)

    P_test = len(torch.unique(test_x[:, id_covariate]))

    if nnet_model.conv and test:
        indexes = np.concatenate([np.array(range(5, T)) + i * T for i in range(P_test)])
    elif test:
        indexes = np.array(list(set(np.array(test_x[:, -1].cpu(), int)) - set(training_indexes)), int)

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_test):
            label = sample_batched['label'].to(device)

            if test:
                if nnet_model.conv:
                    test_indexes = list(set.intersection(set(np.array(sample_batched['idx'])), set(indexes)))
                    data = sample_batched['digit'][[i in test_indexes for i in list(np.array(sample_batched['idx']))], :].to(device)
                    mask = sample_batched['mask'][[i in test_indexes for i in list(np.array(sample_batched['idx']))], :].to(device)
                    param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)[
                                 [i in test_indexes for i in list(np.array(sample_batched['idx']))], :]
                else:
                    test_indexes = list(set.intersection(set(np.array(label[:, -1].cpu(), int)), set(indexes)))
                    tensor_indexes = [label[i, -1] in test_indexes for i in range(label.shape[0])]
                    data = sample_batched['digit'][tensor_indexes,:].to(device)
                    mask = sample_batched['mask'][tensor_indexes,:].to(device)
                    param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)[tensor_indexes, :]
            else:
                data = sample_batched['digit'].to(device)
                mask = sample_batched['mask'].to(device)
                param_mask = sample_batched['param_mask'].view(sample_batched['param_mask'].shape[0], -1)
                
            param_mask = param_mask.to(device).to(torch.float64)
            data = torch.squeeze(data).to(torch.float64)
            mask = torch.squeeze(mask).to(torch.float64)

            _, _, _, params_x_test, log_p_x_test, log_p_x_test_missing = \
                nnet_model.get_test_samples(data, mask, param_mask)

            p_params_test_list.append(params_x_test)
            log_p_x_test_list.append(log_p_x_test)
            log_p_x_test_missing_list.append(log_p_x_test_missing)

        # Determine tensor indexes for data extraction
        if test and not nnet_model.conv:
            tensor_indexes = [test_dataset.label_source.values[i, -1] in indexes for i in range(test_dataset.label_source.shape[0])]
        else:
            tensor_indexes = indexes

        # Get reconstructed/imputed data
        data_source_dev = torch.tensor(test_dataset.data_source.values[tensor_indexes,:]).to(torch.float64).to(device)
        mask_source_dev = torch.tensor(test_dataset.mask_source.values[tensor_indexes,:]).to(torch.float64).to(device)
        
        p_params_complete = read_functions.p_params_concatenation_by_key(p_params_test_list, test_dataset.types_info,
                                                                      len(mask_source_dev), data.device, 'x')
        
        # THIS IS WHERE THE IMPUTATION HAPPENS
        recon_batch_mean, recon_batch_mode = read_functions.statistics(p_params_complete, test_dataset.types_info, data.device, log_vy=[nnet_model._log_vy_real, nnet_model._log_vy_pos])

        # Transform original data to same format as reconstructed data
        train_data_transformed = read_functions.discrete_variables_transformation(data_source_dev, test_dataset.types_info)

        # Convert to CPU numpy arrays for saving
        ground_truth = train_data_transformed.cpu().numpy()
        imputed_mean = recon_batch_mean.cpu().numpy()
        imputed_mode = recon_batch_mode.cpu().numpy()
        mask_np = mask_source_dev.cpu().numpy()

        # Create comparison DataFrame
        n_samples, n_features = ground_truth.shape
        
        # Get feature names (assuming they match your Y columns)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        if hasattr(test_dataset, 'Y_cols'):
            feature_names = test_dataset.Y_cols[:n_features]
        
        comparison_data = []
        
        for sample_idx in range(n_samples):
            for feat_idx in range(n_features):
                comparison_data.append({
                    'sample_id': sample_idx,
                    'feature': feature_names[feat_idx] if feat_idx < len(feature_names) else f'feature_{feat_idx}',
                    'ground_truth': ground_truth[sample_idx, feat_idx],
                    'imputed_mean': imputed_mean[sample_idx, feat_idx], 
                    'imputed_mode': imputed_mode[sample_idx, feat_idx],
                    'was_missing': 1 - mask_np[sample_idx, feat_idx],  # 1 if missing, 0 if observed
                    'absolute_error_mean': abs(ground_truth[sample_idx, feat_idx] - imputed_mean[sample_idx, feat_idx]),
                    'absolute_error_mode': abs(ground_truth[sample_idx, feat_idx] - imputed_mode[sample_idx, feat_idx])
                })
        
        # Save to CSV
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(save_path, index=False)
        print(f"Ground truth vs imputed data saved to: {save_path}")
        
        # Also save a summary of missing vs observed errors
        missing_data = df_comparison[df_comparison['was_missing'] == 1]
        observed_data = df_comparison[df_comparison['was_missing'] == 0]
        
        print("\n=== IMPUTATION PERFORMANCE SUMMARY ===")
        if len(missing_data) > 0:
            print(f"Missing data points: {len(missing_data)}")
            print(f"Mean absolute error (mean imputation): {missing_data['absolute_error_mean'].mean():.4f}")
            print(f"Mean absolute error (mode imputation): {missing_data['absolute_error_mode'].mean():.4f}")
        
        if len(observed_data) > 0:
            print(f"Observed data points: {len(observed_data)}")
            print(f"Mean absolute error (mean reconstruction): {observed_data['absolute_error_mean'].mean():.4f}")
            print(f"Mean absolute error (mode reconstruction): {observed_data['absolute_error_mode'].mean():.4f}")

    # Continue with original function logic for other metrics...
    # [Rest of the original function code for computing other metrics]
    
    return df_comparison  # Return the comparison DataFrame as well