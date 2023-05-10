
import filecmp
from utils.dataloader import DataLoader

from getpass import getpass
import os
from turtle import down
import unittest



class LoaderTest(unittest.TestCase):
    
    def test_loader(self):
        nc_user = input("NextCloud username:")
        nc_pass = getpass("NextCloud password:")
        loader = DataLoader(
                    ml_user='test', ml_pass='test',
                    nc_user=nc_user, nc_pass=nc_pass,
                    ml_platform_address = 'ml-platform.test.rd.nask.pl',
                    ml_keycloak_address = 'auth.rd.nask.pl',
                    ml_realm  = 'ML-PLATFORM',
                    ml_client_id  = 'ml_platform',
                    nc_host  = 'https://nextcloud.cbitt.nask.pl/'
                )


        remote_test_file_path = loader.UPLOAD_DIRECTORY + "/upload_test.csv"
        local_test_file_path = "upload_test.csv"
        loader.DOWNLOAD_DIRECTORY = "download_testdir/"

        downloaded_file_path = "download_testdir/upload_test.csv"
        to_del_paths = []
        print(f"Uploading test file: {local_test_file_path}")
        self.assertTrue(loader.upload(local_test_file_path))
        print(f"and downloading it to directory: {loader.DOWNLOAD_DIRECTORY}")
        self.assertTrue(loader.download(remote_test_file_path))

        self.assertTrue(os.path.isfile(downloaded_file_path))

        downloaded_file_path = "download_testdir/other_filename.csv"
        print("Downloading with new filename: other_filename.csv")
        self.assertTrue(loader.download(remote_test_file_path, local_filename="other_filename.csv"))
        self.assertTrue(os.path.isfile(downloaded_file_path))
        
        to_del_paths.append(downloaded_file_path)

        print("Trying to upload again the same file")
        self.assertTrue(loader.upload(local_test_file_path))

        print("Uploading the same file and registering")
        upload_result = loader.upload_and_register(os.path.abspath(local_test_file_path))
        self.assertIsNotNone(upload_result)
        downloaded_file_path = "download_testdir/registered_filename.csv"
        print("Downloading registered file")
        registered_download =\
            loader.download_registered(upload_result, local_filename="registered_filename.csv")

        self.assertIsNotNone(registered_download)
        self.assertTrue(os.path.isfile(downloaded_file_path))
        to_del_paths.append(downloaded_file_path)

        for path in to_del_paths:
            self.assertTrue(filecmp.cmp(path, local_test_file_path))


        loader.delete_downloaded()
        
        os.remove(os.path.abspath(loader.DOWNLOAD_DIRECTORY))
        
    