wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
mkdir fma_small_data
unzip -q fma_small.zip -d fma_small_data
unzip -j fma_metadata.zip fma_metadata/tracks.csv -d fma_small_data
pip install -r requirements.txt
