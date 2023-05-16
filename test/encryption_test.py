# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


from fedstellar.communication_protocol import CommunicationProtocol
from fedstellar.encrypter import AESCipher
from fedstellar.encrypter import RSACipher
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.learning.pytorch.mnist.models.mlp import MLP
from test.utils import set_test_settings

set_test_settings()


#############################
#    RSA Encryption Test    #
#############################


def test_rsa_encryption_decription1():
    rsa = RSACipher()
    rsa.load_pair_public_key(rsa.get_key())
    message = "Hello World!".encode("utf-8")
    encrypted_message = rsa.encrypt(message)
    decrypted_message = rsa.decrypt(encrypted_message)
    assert message == decrypted_message


def test_rsa_encryption_decription2():
    cipher1 = RSACipher()
    cipher2 = RSACipher()
    cipher1.load_pair_public_key(cipher2.get_key())
    cipher2.load_pair_public_key(cipher1.get_key())
    # Exchange messages and check if they are the same
    message = "Buenas tardes Dani y Enrique!".encode("utf-8")
    encrypted_message1 = cipher1.encrypt(message)
    encrypted_message2 = cipher2.encrypt(message)
    decrypted_message1 = cipher2.decrypt(encrypted_message1)
    decrypted_message2 = cipher1.decrypt(encrypted_message2)
    assert message == decrypted_message1
    assert message == decrypted_message2


#############################
#    AES Encryption Test    #
#############################


def test_aes_encryption_decription1():
    cipher = AESCipher()
    message = "zzZZZZ!"
    encoded_mesage = cipher.add_padding(message.encode("utf-8"))
    encrypted_message = cipher.encrypt(encoded_mesage)
    decrypted_message = cipher.decrypt(encrypted_message)
    decoded_message = " ".join(decrypted_message.decode("utf-8").split())
    assert message == decoded_message


def test_aes_encryption_decription2():
    cipher1 = AESCipher()
    cipher2 = AESCipher(key=cipher1.get_key())
    message = "balb l    ablab alb  a lbalabla     bal"
    encoded_mesage = cipher1.add_padding(message.encode("utf-8"))
    encrypted_message = cipher1.encrypt(encoded_mesage)
    decrypted_message = cipher2.decrypt(encrypted_message)
    decoded_message = decrypted_message.decode("utf-8")
    assert message.split() == decoded_message.split()


def test_aes_encryption_decription_model():
    cipher = AESCipher()
    nl = LightningLearner(MLP(), None)
    encoded_parameters = nl.encode_parameters()

    messages = CommunicationProtocol.build_params_msg(encoded_parameters)

    encrypted_messages = [cipher.encrypt(cipher.add_padding(x)) for x in messages]
    decrypted_messages = [cipher.decrypt(x) for x in encrypted_messages]

    for i in range(len(messages)):
        assert messages[i] == decrypted_messages[i]
