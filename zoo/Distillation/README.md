# Distillation

В папке lib находятся файлы модели BlendCNN  и код для обучения модели с помощью дистилляции (`train_student.py`). Для того, чтобы всё заработало нужно дополнить папку lib и остальными файлами (bert.py, train_eval.py и прочими). 

В папке notebooks находятся эксперименты. 

`simple_distillation_example.ipynb` - простой пример, показывающий как пользоваться дистилляцией.

`train_student_bert_classifier.ipynb` содержит эксперимент с обучением обрезаного до 6 блоков берта, а также эксперимент с важносьтью слоев.

`train_blend_cnn.ipynb` содержит эксперимент с обучением модели BlendCNN при помощи дистиляции.

`train_blend_cnn_only.ipynb` - обучение BlendCNN без дистиляции на задаче SST-2.

Пока что все эксперименты проводились со старой версией lilbert.