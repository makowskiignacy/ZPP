

import os

from mlplatformclient import MLPlatformClient, NextCloud

class DataLoader:
    UPLOAD_DIRECTORY = "/uw_demo"
    DOWNLOAD_DIRECTORY = "downloads/"

    def __init__(self,
                ml_user : str, ml_pass : str,
                nc_user : str, nc_pass : str,
                # TODO ustawić 'lepsze' wartości domyślne
                ml_platform_address : str = 'ml-platform.test.rd.nask.pl',
                ml_keycloak_address: str = 'auth.rd.nask.pl',
                ml_realm : str = 'ML-PLATFORM',
                ml_client_id : str = 'ml_platform',
                nc_host : str = 'https://nextcloud.cbitt.nask.pl/'
                 ) -> None:

        
        self.__next_cloud = NextCloud(nc_host, nc_user, nc_pass)
        self.__ml_platform_client = MLPlatformClient(
                                        ml_platform_address,
                                        ml_keycloak_address,
                                        ml_realm,
                                        ml_client_id,
                                        ml_user,
                                        ml_pass
                                    )
    
    def upload_and_register(self, local_path : str, remote_dir : str | None = None) :
        """
        Przesyła plik wskazany przez local_path do NextCloud w miejsce określone
        przez remote_dir lub ustawione wcześniej UPLOAD_DIRECTORY. Następnie
        rejestruje podany plik w ML_Platform.

        Na wyjściu przekazuje wynik rejestracji lub None jeżeli przesyłanie
        lub rejestracja nie powiodły się
        """
        if not self.upload(local_path, remote_dir):
            return None

        if remote_dir is None:
            remote_dir = self.UPLOAD_DIRECTORY

        remote_path = os.path.join(remote_dir, os.path.basename(local_path))
        upload_result = self.__ml_platform_client.register_dataset_and_check_upload(remote_path)

        if upload_result == False:
            return None

        return upload_result

    def download_registered(self,
                            upload_result : dict | None, local_filename : str,
                            local_download_path : str | None = None) -> dict | None:
        """
        Pobiera plik identyfikowany poprzez wynik jego rejestracji 'upload_result'
        w miejsce wskazane przez 'local_download_path' i ustawia nazwę na
        'local_filename'

        W wyniku przekazuje słownik opisujący nazwę i ścieżkę do pliku
        {'file_name': name, 'file_path': file_path}. 
        Jeżeli rejestracja niepowiodła się ('upload_results' jest None) przekazuje
        na wyjściu None.
        """
        if upload_result is None:
            return None

        if local_download_path is None:
            local_download_path = self.DOWNLOAD_DIRECTORY
        
        return self.__ml_platform_client.get_dataset(upload_result['id'], local_filename, local_download_path)
    
    def upload(self, local_path : str, remote_dir : str | None = None) -> bool:
        """
        Przesyła wskazany przez 'local_path' plik do katalogu 'remote_dir'
        jeżeli podano lub do katalogu UPLOAD_DIRECTORY
        UWAGA: Ta procedura nadpisze istniejący plik.
        """

        if remote_dir is None:
            remote_dir = self.UPLOAD_DIRECTORY

        local_absolute_path = os.path.abspath(local_path)
        return self.__next_cloud.upload_file(local_absolute_path, remote_dir)

    def download(self,
                 remote_path : str,
                 local_download_path : str | None = None,
                 local_filename : str | None = None) -> bool:
        """
        Pobiera podany plik spod 'remote_path' do wskazanego przez 'local_download_path'
        katalogu, po czym zmienia jego nazwę na 'local_filename' jeżeli podano.
        UWAGA: Ta procedura nadpisze plik o oryginalnej nazwie!
        """
        if local_download_path is None:
            local_download_path = self.DOWNLOAD_DIRECTORY

        if self.__next_cloud.download_file(remote_path, local_download_path):
            if local_filename is not None:
                downloaded_file_path = os.path.join(os.path.abspath(local_download_path), os.path.basename(remote_path))
                requested_file_path = os.path.join(os.path.abspath(local_download_path), local_filename)
                os.rename(downloaded_file_path, requested_file_path)
            return True
        return False
        
