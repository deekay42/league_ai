import time
import threading
import importlib
sstarttime = time.time()

cv = None
Item = None
MinMaxScaler = None


def setopencv(val):
    global cv
    cv = val

def setcassItem(val):
    global Item
    Item = val.Item

def setsklearnItem(val):
    global MinMaxScaler
    MinMaxScaler = val.MinMaxScaler


def heavy_import(module_name, setter):
    result = importlib.import_module(module_name)
    setter(result)
    print(f"Thread {module_name} OVERALL Took {time.time() - sstarttime} s")
    
    
def sequence_imports():
    heavy_import("sklearn.preprocessing.data", setsklearnItem)
    heavy_import("cassiopeia.core.staticdata.item", setcassItem)

threading.Thread(target=lambda:heavy_import("cv2", setopencv)).start()
threading.Thread(target=sequence_imports).start()