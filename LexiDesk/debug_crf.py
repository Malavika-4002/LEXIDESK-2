import torchcrf
import inspect

print(f"File: {torchcrf.__file__}")
try:
    print(f"Init signature: {inspect.signature(torchcrf.CRF.__init__)}")
except Exception as e:
    print(f"Could not get signature: {e}")

try:
    print(f"CRF doc: {torchcrf.CRF.__doc__}")
except:
    pass
