import unittest

from src.evaluation.thai_text import edit_distance, error_rates, normalize_thai, tokenize_words


class NormalizeThaiTest(unittest.TestCase):
    def test_punctuation_and_whitespace_removed(self):
        self.assertEqual(normalize_thai("  สวัสดี ครับ!  “ทดสอบ”, (ok) ๚ "), "สวัสดีครับทดสอบok")

    def test_digits_converted_consistently(self):
        words = normalize_thai("ปี สองพันห้าร้อยหกสิบสี่")
        self.assertEqual(normalize_thai("ปี 2564"), words)
        self.assertEqual(normalize_thai("ปี ๒๕๖๔"), words)
        self.assertEqual(normalize_thai("ปี 2,564"), words)
        self.assertEqual(normalize_thai("3.25"), "สามจุดสองห้า")
        self.assertEqual(normalize_thai("１２"), normalize_thai("12"))  # full-width digits (NFKC)

    def test_unicode_variants_agree(self):
        self.assertEqual(normalize_thai("น้ํา"), normalize_thai("น้ำ"))  # decomposed SARA AM
        self.assertEqual(normalize_thai("เเปลก"), normalize_thai("แปลก"))  # two SARA E typed as SARA AE
        self.assertEqual(normalize_thai("Hello"), "hello")

    def test_mai_yamok_kept(self):
        self.assertEqual(normalize_thai("เด็ก ๆ"), "เด็กๆ")


class ErrorRateTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)
        self.assertEqual(edit_distance(["a", "b"], ["a", "b"]), 0)

    def test_identical_after_normalisation_is_zero(self):
        rates = error_rates("ขอให้ทุกคน มีสุขภาพแข็งแรง", "ขอให้ทุกคนมีสุขภาพแข็งแรง!")
        self.assertEqual((rates["cer"], rates["wer"]), (0.0, 0.0))

    def test_cer_and_wer_values(self):
        rates = error_rates("ไม่ว่าจะเสนออะไร", "ไม่ว่าจะเสนอ")
        self.assertEqual(rates["cer_ref_len"], len("ไม่ว่าจะเสนออะไร"))
        self.assertEqual(rates["cer_edits"], len("อะไร"))
        self.assertEqual(tokenize_words(rates["ref_norm"]), ["ไม่", "ว่า", "จะ", "เสนอ", "อะไร"])
        self.assertEqual((rates["wer_edits"], rates["wer_ref_len"]), (1, 5))

    def test_empty_hypothesis_is_all_deletions(self):
        rates = error_rates("สวัสดี", "")
        self.assertEqual((rates["cer"], rates["wer"]), (1.0, 1.0))

    def test_insertions_can_exceed_one(self):
        self.assertGreater(error_rates("ดี", "ดีมากมากมากมาก")["cer"], 1.0)


if __name__ == "__main__":
    unittest.main()
