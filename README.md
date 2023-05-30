# ZPP

### Trello
https://trello.com/w/zpp58

### Troubleshooting Document
https://docs.google.com/spreadsheets/d/1DY7JFo9h3nN2REfxSgCWGac-TPBXbPjWFAGx5hqhdkY/edit?usp=sharing

### Naming convention

https://realpython.com/python-pep8/#naming-styles

### Attack architecture

https://drive.google.com/file/d/1ATXzWtryI_7qKLSlHj5B1tvl2KJ356eF/view

![BasicInterface](https://user-images.githubusercontent.com/78569836/206575611-cab8fb0b-f817-499e-9232-30b4574bea1c.png)

## Konfiguracja testów
cd \<project-directory>

python3 -m unittests
### Wykonanie testu:
python3 -m unittest.tests.\<nazwa-odpowiedniej-klasy>

### Wykonanie wszystkich unittestów:
python3 -m unittest discover

### Cloud config
W pliku `/utils/cloud_config.py` należy podać swój login oraz hasło, by umożliwić połączenie z zasobami NASK.

### Docker POC:
Umieścić plik `Internal_CBiTT_CA_.crt` w głównym katalogu.\
W pliku `.env` należy podać swój login oraz hasło w zmiennych `CLOUD_USERNAME` i `CLOUD_PASSWORD`, by umożliwić połączenie z zasobami NASK oraz dodoać zmienną: `GIT_PYTHON_REFRESH=quiet`\

Uruchomienie:\
`docker build -t <img_name>`\
`docker run --env-file=<env_file> <img_name>`\
By zatrzymać:\
`docker ps`\
`docker stop <container_id>\
