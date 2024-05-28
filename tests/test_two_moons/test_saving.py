
import keras

from tests.utils import assert_layers_equal


def test_save_and_load(tmp_path, amortizer):
    amortizer.save(tmp_path / "amortizer.keras")
    loaded_amortizer = keras.saving.load_model(tmp_path / "amortizer.keras")

    assert_layers_equal(amortizer, loaded_amortizer)
