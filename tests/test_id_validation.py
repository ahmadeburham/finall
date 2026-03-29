from idv.modules.normalization import validate_id_number


def test_id_valid():
    ok, _ = validate_id_number("29801010101010", [14])
    assert ok


def test_id_bad_length():
    ok, meta = validate_id_number("123", [14])
    assert not ok
    assert meta["reason"] == "bad_length"
