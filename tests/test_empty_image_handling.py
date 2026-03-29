from pathlib import Path

import pytest

from idv.modules.image_io import ImageLoadError, load_color_image


def test_empty_image_handling(tmp_path: Path):
    p = tmp_path / "missing.png"
    with pytest.raises(ImageLoadError):
        load_color_image(p)
