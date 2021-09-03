
start 15 fps Копировать память device->host->device
python3.7 test_write_video.py -i 1_min_15fps.mp4 -f 15 --out_format mp4 -mem_tr True

start 30 fps out 30 fps # Зеленые кадрые 
python3.7 test_write_video.py -i 1_min_30fps.mp4 -f 30 --out_format mp4 -mem_tr True

start 30 fps out 15 fps # Нет зеленых  кадров 
python3.7 test_write_video.py -i 1_min_30fps.mp4 -f 30 --out_format mp4 -mem_tr True -cut True

Видео файл будет в $PWD/HSLINK 
Папка очищается каждый запуск