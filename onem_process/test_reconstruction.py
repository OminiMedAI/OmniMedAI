"""Tests for reusable super-resolution reconstruction interfaces."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import scipy  # noqa: F401

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class TestInterpolationReconstruction(unittest.TestCase):
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "numpy or scipy unavailable")
    def test_cubic_array_reconstruction(self):
        from onem_process import InterpolationReconstructor, ReconstructionConfig

        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = InterpolationReconstructor(
            ReconstructionConfig(scale_factors=(2, 3), interpolation="cubic")
        ).reconstruct_array(image)
        self.assertEqual(result.image.shape, (8, 12))
        self.assertEqual(result.metadata["algorithm"], "interpolation:cubic")
        self.assertEqual(result.metadata["scale_factors"], [2, 3])


class TestReconstructionRegistry(unittest.TestCase):
    def test_lists_common_super_resolution_algorithms(self):
        from onem_process import list_reconstruction_algorithms

        algorithms = list_reconstruction_algorithms()
        for name in ["srgan", "rdgan", "swin2sr", "hat", "custom_torch"]:
            self.assertIn(name, algorithms)

    def test_srgan_4x_mri_preset_records_public_parameters(self):
        from onem_process import srgan_4x_mri_config

        config = srgan_4x_mri_config(
            checkpoint_path="restricted/srgan_generator.pt",
            batch_size=16,
        )
        self.assertEqual(config.algorithm, "srgan")
        self.assertEqual(config.scale_factors, (4.0, 4.0, 1.0))
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.model_parameters["mode"], "slice_wise")
        self.assertEqual(config.model_parameters["normalization"], "z_score")

    def test_multiple_4x_mri_presets_are_exported(self):
        from onem_process import (
            RECONSTRUCTION_PRESETS,
            custom_torch_4x_mri_config,
            esrgan_4x_mri_config,
            hat_4x_mri_config,
            rdgan_4x_mri_config,
            srgan_4x_mri_config,
            swin2sr_4x_mri_config,
            swinir_4x_mri_config,
        )

        preset_functions = {
            "custom_torch_4x_mri": custom_torch_4x_mri_config,
            "esrgan_4x_mri": esrgan_4x_mri_config,
            "hat_4x_mri": hat_4x_mri_config,
            "rdgan_4x_mri": rdgan_4x_mri_config,
            "srgan_4x_mri": srgan_4x_mri_config,
            "swin2sr_4x_mri": swin2sr_4x_mri_config,
            "swinir_4x_mri": swinir_4x_mri_config,
        }
        self.assertEqual(set(preset_functions), set(RECONSTRUCTION_PRESETS))
        for name, factory in preset_functions.items():
            config = factory(checkpoint_path=f"weights/{name}.pt")
            self.assertEqual(config.scale_factors, (4.0, 4.0, 1.0))
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.model_parameters["mode"], "slice_wise")
            self.assertEqual(config.model_parameters["output_format"], "nifti")


if __name__ == "__main__":
    unittest.main()
