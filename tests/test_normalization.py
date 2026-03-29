from idv.modules.normalization import normalize_digits


def test_normalize_digits_arabic_and_latin():
    assert normalize_digits("رقم ١٢٣4٥") == "12345"
