

import itertools
import os
import csv
import torch
from enum import Enum

from mlplatformclient import MLPlatformClient, NextCloud
from attacks.helpers.data import Data



class DataLoader:
    '''
    Klasa obiektów implementujących zarządzanie pobieraniem danych oraz modeli
    z platform NextCloud oraz ML Platform.
    '''
    class SupportedLibrary(Enum):
        ART = 1
        Foolbox = 2
    UPLOAD_DIRECTORY = "/uw_demo"
    DOWNLOAD_DIRECTORY = "downloads/"
    downloaded_files = []

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
        '''
        Inicjalizuje obiekt zarządzający.

        Parametry inicjalizacji:
        ------------------------
        ml_user: (str)
            Nazwa użytkownika do ML Platform
        ml_pass: (str)
            Hasło używkownika do ML Platform
        nc_user: (str)
            Nazwa użytkownika do NextCloud
        nc_pass: (str)
            Hasło użytkownika do NextCloud
        ml_platform_address: (str)
            Adres internetowy usługi ML Platform
        ml_keycloak_address: (str)
            Adres internetowy uwierzytelniający do ML Platform
        ml_realm: (str)
        ml_client_id: (str)
            ID klienta usługi ML Platform
        nc_host: (str)
            Adres internetowy usługi NextCloud
        '''
        
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
        Przesyła wskazany plik do NextCloud, a następnie rejestruje podany plik
        w usłudze ML Platform.

        Miejsce docelowe jest określane na podstawie remote_dir lub
        ustawionego wcześniej UPLOAD_DIRECTORY. 

        Na wyjściu przekazuje wynik rejestracji lub None jeżeli przesyłanie
        lub rejestracja nie powiodły się.

        Parametry:
        ----------
        local_path: (str)
            Ścieżka lokalna do przesyłanego pliku
        remote_dir: (str | None)
            Ścieżka docelowa pliku do przesłania
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

    def download_by_id(self, id, local_filename : str, local_download_path : str | None = None) :
        if local_download_path is None:
            local_download_path = self.DOWNLOAD_DIRECTORY
        
        retval = self.__ml_platform_client.get_dataset(id, local_filename, local_download_path)
        file_path = os.path.join(local_download_path, local_filename)
        self.downloaded_files.append(file_path)
        return retval
        
    def download_registered(self,
                            upload_result : dict | None, local_filename : str,
                            local_download_path : str | None = None) -> dict | None:
        """
        Pobiera plik identyfikowany poprzez wynik jego rejestracji do miejsca
        wskazanego poprzez parametr lub wcześniej ustawione UPLOAD_DIRECTORY.

        W wyniku przekazuje słownik opisujący nazwę i ścieżkę do pliku
        {'file_name': name, 'file_path': file_path}. 

        Jeżeli rejestracja niepowiodła się ('upload_results' jest None)
        przekazuje na wyjściu None.

        Parametry:
        ----------
        upload_result: (dict | None)
            Wynik rejestracji pliku otrzymany z metody 'upload_and_register()'
        local_filename: (str)
            Nazwa pliku ustawiana po pobraniu
        local_download_path: (str | None)
            Ścieżka do katalogu, do którego ma zostać pobrany plik
        """
        if upload_result is None:
            return None

        return self.download_by_id(upload_result['id'], local_filename, local_download_path)
    
    def upload(self, local_path : str, remote_dir : str | None = None) -> bool:
        """
        Przesyła wskazany plik do NextCloud.
        UWAGA: Ta procedura nadpisze istniejący plik.

        Parametry:
        ----------
        local_path: (str)
            Ścieżka lokalna do przesyłanego pliku
        remote_dir: (str | None)
            Ścieżka docelowa pliku do przesłania
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
        Pobiera podany plik pod wskazane przez parametr lub wcześniej ustawione
        DOWNLOAD_DIRECTORY miejsce.
        UWAGA: Ta procedura nadpisze plik o oryginalnej nazwie!

        Parametry:
        ----------
        remote_path: (str)
            Ścieżka do pobieranego pliku
        local_filename: (str | None)
            Nowa nazwa pliku ustawiana po pobraniu.
        local_download_path: (str | None)
            Ścieżka do katalogu, do którego ma zostać pobrany plik
        """
        if local_download_path is None:
            local_download_path = self.DOWNLOAD_DIRECTORY
        
        if self.__next_cloud.download_file(remote_path, local_download_path):
            downloaded_file_path = os.path.join(os.path.abspath(local_download_path),
                                                os.path.basename(remote_path))
            if local_filename is not None:
                
                requested_file_path = os.path.join(os.path.abspath(local_download_path), local_filename)
                os.rename(downloaded_file_path, requested_file_path)
                self.downloaded_files.remove(downloaded_file_path)
                self.downloaded_files.append(requested_file_path)
            else:
                self.downloaded_files.append(downloaded_file_path)
            return True
        return False

    def load_to_variable(self, libs : list[SupportedLibrary], local_file_path : str, number_of_samples : int | None = None, column_names_row : int = 0, labels_column_number : int | None = None):
        '''
        Ładuje dane zawarte w wskazanym pliku do zmiennej wyjściowej w formacie
        kompatybilnym z resztą modułu.

        Parametry:
        ----------
        libs: (list[SupportedLibrary])
            Lista enumetatorów wspieranych bibliotek. W wyjściowej liście
            pojawią się dane załadowane w odpowiednim formacie dla odpowiedniej
            biblioteki w kolejności podanej na liście.
        local_file_path: (str)
            Ścieżka do pliku z danymi
        number_of_samples: (int | None)
            Liczba próbek do wyloswania z danych i następnego załadowania.
            Wartość 'None' odpowiada załadowaniu wszystkich danych.
        column_names_row: (int)
            Numer wiersza z opisami nazw kolumn.
        labels_column_number: (int)
            Numer kolumny z wyjściowymi etykietami.

        '''
        with open(local_file_path) as f:
            reader = csv.reader(f)
            if number_of_samples is not None:
                data = list(list(line) for line in itertools.islice(reader, number_of_samples))
            else:
                data = list(list(line) for line in reader)
                # NOTE czy tutaj chcemy robić ten preprocessing?
                # Niebrałem udziału w 'ładowniu danych' stąd nie do
                # końca wiem czy coś tu jeszcze by się przydało

                # Zakładamy domyślnie że pierwszy wiersz to etykiety
            data.pop(column_names_row)

        data2 = []
        for row in data:
            row2 = []
            for place in range(len(row)):
                row2.append(float(row[place]))
            data2.append(row2)
        data = data2
        data = torch.tensor(data, requires_grad=False, dtype=torch.float)
        
        number_of_columns = len(data[0])
        if labels_column_number is None:
            # Zakładamy domyślnie, że wynik są ostatnią kolumną
            data, labels = torch.hsplit(data, [number_of_columns - 1, ])
            labels = torch.tensor(labels, requires_grad=False, dtype=torch.float)
        else:
            data1, data2 = torch.hsplit(data, [0, labels_column_number])
            labels, data2 = torch.hsplit(data2, [1,])
            data = torch.hstack([data1, data2])
            data = torch.tensor(data, requires_grad=False, dtype=torch.float)
            labels = torch.tensor(labels, requires_grad=False, dtype=torch.float)
            
        retlist = []
        for lib in libs:
            if lib == DataLoader.SupportedLibrary.ART:
                outdata = Data(data.numpy(), labels.numpy())
            elif lib == DataLoader.SupportedLibrary.Foolbox:
                outdata = Data(data, labels)
            else:
                raise NotImplementedError(f"{lib.name} is not yet supported.")
            retlist.append(outdata)
            
        return retlist

    def delete_downloaded(self):
        '''
        Usuwanie wszystkich pobranych do tej pory plików.
        '''
        for path in self.downloaded_files:
            os.remove(path)

    def make_remote_path(self, remote_file : str):
        '''
        Procedura pomocnicza tworząca ścieżkę do pliku zdalnego.
        Przekazuje: UPLOAD_DIRECTORY/remote_file
        '''
        return os.path.join(self.UPLOAD_DIRECTORY, remote_file)
    
    def make_local_path(self, local_file : str):
        '''
        Procedura pomocnicza tworząca ścieżkę do pliku lokalnego.
        Przekazuje: DOWNLOAD_DIRECTORY/local_file
        '''
        return os.path.join(self.DOWNLOAD_DIRECTORY, local_file)
