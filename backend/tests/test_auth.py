import jwt
import pytest

from app.auth import AuthError, create_token, decode_token, hash_password, verify_password

SECRET = "unit-test-secret-at-least-32-bytes-long"


def test_password_roundtrip():
    h = hash_password("correct horse battery")
    assert h != "correct horse battery"
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong password", h)


def test_long_passwords_are_fully_significant():
    base = "a" * 100
    h = hash_password(base + "X")
    assert not verify_password(base + "Y", h)  # differ only after byte 72


def test_hash_is_salted():
    assert hash_password("same-password") != hash_password("same-password")


def test_verify_handles_garbage_hash():
    assert verify_password("pw", "not-a-bcrypt-hash") is False


def test_token_roundtrip():
    assert decode_token(create_token(42, SECRET), SECRET) == 42


def test_wrong_secret_rejected():
    with pytest.raises(AuthError):
        decode_token(create_token(1, SECRET), SECRET + "-other")


def test_expired_token_rejected():
    with pytest.raises(AuthError):
        decode_token(create_token(1, SECRET, minutes=-1), SECRET)


def test_token_without_subject_rejected():
    token = jwt.encode({"exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(token, SECRET)
