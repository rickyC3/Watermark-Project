from .watermark import WatermarkEncoder, WatermarkDecoder

'''
在電腦裡，資料夾就只是個裝檔案的容器。但當你在某個資料夾（例如 imwatermark）裡面放進一個叫做 __init__.py 的檔案時，就像是對 Python 施了魔法。

Python 只要看到這個檔案，就會立刻認定：「喔！這不是一個普通的 Windows 資料夾，這是一個正式的 Python 工具箱（Package/套件），我可以允許別人 import 它！」

如果沒有這個檔案，當你在腳本裡寫 import imwatermark 時，Python 會直接瞎掉，跟你說找不到這個套件。

'''