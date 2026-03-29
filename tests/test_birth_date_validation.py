from idv.modules.normalization import validate_birth_date


def test_birth_date_valid():
    ok, meta = validate_birth_date("31/12/2000", 1900, 2100)
    assert ok
    assert meta["formatted"] == "2000-12-31"


def test_birth_date_invalid():
    ok, meta = validate_birth_date("99/99/2000", 1900, 2100)
    assert not ok
    assert meta["reason"] == "invalid_date"
