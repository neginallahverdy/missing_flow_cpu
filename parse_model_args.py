import argparse
import ast


"""
Helper code for loading parameters from parameter file or from command line
"""

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)


class ModelArgs:
    """
    Runtime parameters for the L-VAE model
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the model')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--csv_file_prediction_data', type=str, help='Name of prediction data file', required=False)
        self.parser.add_argument('--csv_file_prediction_label', type=str, help='Name of prediction label file', required=False)
        self.parser.add_argument('--csv_types_file', type=str, help='Name of types file', required=False)
        self.parser.add_argument('--true_mask_file', type=str, help='Name of true mask file', default='')
        self.parser.add_argument('--true_test_mask_file', type=str, help='Name of true mask file for test data', default='')
        self.parser.add_argument('--true_prediction_mask_file', type=str, help='Name of true mask file for prediction data', default='')
        self.parser.add_argument('--true_validation_mask_file', type=str, help='Name of true mask file for validation data', default='')
        self.parser.add_argument('--true_generation_mask_file', type=str, help='Name of true mask file for generation data', default='')
        self.parser.add_argument('--csv_file_validation_data', type=str, help='Name of validation data file', required=False)
        self.parser.add_argument('--csv_file_validation_label', type=str, help='Name of validation label file', required=False)
        self.parser.add_argument('--csv_file_generation_data', type=str, help='Name of data file for image generation', required=False)
        self.parser.add_argument('--csv_file_generation_label', type=str, help='Name of label file for image generation', required=False)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--test_mask_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--prediction_mask_file', type=str, help='Name of prediction mask file', default=None)
        self.parser.add_argument('--validation_mask_file', type=str, help='Name of validation mask file', default=None)
        self.parser.add_argument('--generation_mask_file', type=str, help='Name of mask file for image generation', default=None)
        self.parser.add_argument('--csv_range_file', type=str, help='Name of types file', required=False)
        self.parser.add_argument('--dataset_type', required=False,
                                 choices=['RotatedMNIST', 'HealthMNIST', 'Physionet', 'Physionet2019', 'HeteroHealthMNIST', 'PPMI', 'CustomWeather'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for RNN')
        self.parser.add_argument('--hidden_layers', type=str, help='Number of hidden dimensions for hidden layers')
        self.parser.add_argument(
            '--id_covariate',
            type=int,
            default=0,
            help='Index of ID (unique identifier) covariate')
        self.parser.add_argument('--M', type=int, help='Number of inducing points')
        self.parser.add_argument('--P', type=int, help='Number of unique instances')
        self.parser.add_argument('--T', type=int, help='Number of longitudinal samples per instance')
        self.parser.add_argument('--varying_T', type=str2bool, default=False, help='Varying number of samples per instance')
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--weight', type=float, default=1,
                                 help='Trade-off parameter balancing data reconstruction and latent space prior' +
                                      ' regularisation')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--y_dim', type=int, help='Number of Y dimensions', required=False)
        self.parser.add_argument('--num_samples', type=int, default=1, help='Number of Monte Carlo samples')
        self.parser.add_argument('--type_KL', required=False, choices=['closed', 'other', 'GPapprox', 'GPapprox_closed'],
                                 help='Type of loss computation')
        self.parser.add_argument('--constrain_scales', type=str2bool, default=False, required=False,
                                 help='Constrain the marginal variances')
        self.parser.add_argument('--model_params', type=str, default='model_params.pth',
                                 help='Pre-trained VAE parameters')
        self.parser.add_argument('--gp_model_folder', type=str, default='./pretrainedVAE',
                                 help='Pre-trained GP model parameters')
        self.parser.add_argument('--generate_plots', type=str2bool, default=False, help='Generate plots')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs.')
        self.parser.add_argument('--test_freq', type=int, default=50, help='Period of computing test MSE.')
        self.parser.add_argument('--cat_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_kernel', type=ast.literal_eval)
        self.parser.add_argument('--sqexp_kernel', type=ast.literal_eval)
        self.parser.add_argument('--cat_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--bin_int_kernel', type=ast.literal_eval)
        self.parser.add_argument('--covariate_missing_val', type=ast.literal_eval)
        self.parser.add_argument('--run_tests', type=str2bool, default=False,
                                 help='Perform tests using the trained model')
        self.parser.add_argument('--run_validation', type=str2bool, default=False,
                                 help='Test the model using a validation set')
        self.parser.add_argument('--generate_images', type=str2bool, default=False,
                                 help='Generate images of unseen individuals')
        self.parser.add_argument('--results_path', type=str, required=False, help='Path to results')
        self.parser.add_argument('--f', type=open, action=LoadFromFile)

        self.parser.add_argument('--variational_inference_training', type=str2bool, default=False, help='Use variational inference training.')
        self.parser.add_argument('--memory_dbg', type=str2bool, default=False, help='Debug memory usage in training')
        self.parser.add_argument('--natural_gradient', type=str2bool, default=True, help='Use natural gradients for parameters m and H')
        self.parser.add_argument('--natural_gradient_lr', type=float, default=0.01, help='Learning rate for variational parameters m and H if natural gradient is used')
        self.parser.add_argument('--subjects_per_batch', type=int, default=20, help='Number of subjects per batch in mini-batching.')
        # Compatibility with older config files that specified `batch_size`
        self.parser.add_argument('--batch_size', dest='subjects_per_batch', type=int,
                                 help='Alias for subjects_per_batch (deprecated)')
        self.parser.add_argument('--save_interval', type=int, default=100, help='The frequency of saving the model and images')

        self.parser.add_argument('--vy_init_real', type=float, default=1.0, help='Initial variance for real y in HLVAE')
        self.parser.add_argument('--vy_init_pos', type=float, default=.5, help='Initial variance for positive y in HLVAE')
        self.parser.add_argument('--logvar_network', type=str2bool, default=False, help='Observation variance for real and positive dimensions are trained in HLVAE')
        self.parser.add_argument('--conv_hivae', type=str2bool, default=False, help='Convolutional HLVAE')
        self.parser.add_argument('--conv_range', type=int, default=255, help='Convolutional Range')
        self.parser.add_argument('--early_stopping', type=str2bool, default=False, help='Early Stopping')
        self.parser.add_argument('--use_ranges', type=str2bool, default=False, help='Indicator for Beta Likelihood')
        self.parser.add_argument('--true_mask_validation_file', default=None, type=str,
                    help='CSV mask with ground‑truth for validation set (optional)')
        self.parser.add_argument('--true_mask_test_file', default=None, type=str,
                    help='CSV mask with ground‑truth for test set (optional)')

# اگر نسخهٔ قدیمی از نام true_miss_file استفاده می‌کند، یک alias بدهید:
        self.parser.add_argument('--true_miss_validation_file', dest='true_mask_validation_file', default=None, type=str)
        self.parser.add_argument('--true_miss_test_file', dest='true_mask_test_file', default=None, type=str)

    def parse_options(self):
        opt = vars(self.parser.parse_args())

        # Basic sanity check for required file arguments when using the
        # tabular CustomWeather dataset.  Without these, the dataset class
        # will receive ``None`` and fail with a cryptic ``TypeError`` when
        # calling ``os.path.exists``.
        if opt.get("dataset_type") == "CustomWeather":
            required_keys = ["csv_file_data", "csv_file_label", "mask_file"]
            missing = [k for k in required_keys if not opt.get(k)]
            if missing:
                missing_str = ", ".join("--" + k for k in missing)
                self.parser.error(
                    f"Missing required argument(s) for CustomWeather dataset: {missing_str}"
                )

            # Additional optional splits are required only when their
            # corresponding features are requested.
            if opt.get("run_tests") or opt.get("generate_images"):
                pred_keys = [
                    "csv_file_prediction_data",
                    "csv_file_prediction_label",
                    "prediction_mask_file",
                ]
                missing_pred = [k for k in pred_keys if not opt.get(k)]
                if missing_pred:
                    missing_str = ", ".join("--" + k for k in missing_pred)
                    self.parser.error(
                        f"Missing required argument(s) for prediction dataset: {missing_str}"
                    )

            if opt.get("generate_images"):
                gen_keys = [
                    "csv_file_generation_data",
                    "csv_file_generation_label",
                    "generation_mask_file",
                ]
                missing_gen = [k for k in gen_keys if not opt.get(k)]
                if missing_gen:
                    missing_str = ", ".join("--" + k for k in missing_gen)
                    self.parser.error(
                        f"Missing required argument(s) for generation dataset: {missing_str}"
                    )

            if opt.get("run_validation"):
                val_keys = [
                    "csv_file_validation_data",
                    "csv_file_validation_label",
                    "validation_mask_file",
                ]
                missing_val = [k for k in val_keys if not opt.get(k)]
                if missing_val:
                    missing_str = ", ".join("--" + k for k in missing_val)
                    self.parser.error(
                        f"Missing required argument(s) for validation dataset: {missing_str}"
                    )

        return opt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
